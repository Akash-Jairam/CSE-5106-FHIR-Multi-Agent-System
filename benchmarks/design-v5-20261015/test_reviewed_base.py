"""Synthetic/offline boundary tests; never reads active run/configuration."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec=importlib.util.spec_from_file_location('completion_guard',str(Path(__file__).with_name('reviewed_v4_guard.py')))
g=importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

class CompletionGuard(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.output=self.root/'output'
        self.output.mkdir()
        self.no_network=patch.object(socket,'getaddrinfo',side_effect=AssertionError('network forbidden'))
        self.no_network.start()
        self.addCleanup(self.no_network.stop)
    def artifact(self):
        payload={'_audit_meta':{'question_id':'q1','audit_artifacts_was_present':False},
                 'trace':[{'role':'tool','content':'answer = "λ"\nprint(answer)'}],
                 'retrieved_fhir_resources':{'Observation':[{'id':'public-demo','value':1}]}}
        raw=json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode()
        digest=hashlib.sha256(raw).hexdigest()
        path=self.output/'audit'/(digest+'.json.gz')
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(gzip.compress(raw,mtime=0))
        row={'question_id':'q1','audit_artifacts':{'version':2,'codec':'json+gzip','path':'audit/'+path.name,
             'sha256':digest,'fields':['trace','retrieved_fhir_resources'],'question_id':'q1'}}
        return row,payload,path
    def client(self):
        def cls(label,mro=None):return {'class':label,'mro':mro or [label,'builtins.object']}
        return {'sdk':cls('openai.lib.azure.AzureOpenAI'),
                'http':cls('openai._base_client.SyncHttpxClientWrapper',['openai._base_client.SyncHttpxClientWrapper','httpx2.Client','builtins.object']),
                'transport':cls('httpx2.HTTPTransport'),'pool':cls('httpcore2.ConnectionPool'),
                'sdk_closed':False,'http_closed':False,'client_id':123,'versions':{'openai':'3.26.1','httpx2':'2.13.1'}}
    def events(self,error=False,counter=1):
        start={'version':1,'pid':123,'ppid':122,'counter':counter,'process_session':'a'*32,
               'started_utc':'2026-10-15T01:00:00+00:00','event':'start','client':self.client(),'fd_count':10}
        end={**start,'event':'end','finished_utc':'2026-10-15T01:00:01+00:00','duration_seconds':1,
             'outcome':'error' if error else 'success','exception_chain':[],
             'usage':{'prompt_tokens':None if error else 11,'completion_tokens':None if error else 2,
                      'total_tokens':None if error else 13,'cached_prompt_tokens':None if error else 0},
             'http_status':None,'request_ids':{}}
        if error:
            end['exception_chain']=[{'type':{'class':'httpx2.ReadTimeout','mro':['httpx2.ReadTimeout','builtins.Exception']},
              'errno':None,'winerror':None,'verify_code':None,
              'flags':{'dns':False,'tls':False,'connect':False,'read':True,'write':False,'pool':False,'timeout':True,'resource_fd':False}}]
        return start,end
    def journal(self,events,tail=b''):
        p=self.output/'provider-events'/('azure-diagnostics-123-'+'a'*32+'.jsonl')
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b''.join(json.dumps(e).encode()+b'\n' for e in events)+tail)
        return p
    def test_completion_includes_terminal_failures_but_rejects_unattempted(self):
        self.assertTrue(g.complete({'latency':None,'error':'API timeout; usage unknown'}))
        self.assertTrue(g.complete({'latency':0.0,'error':None}))
        self.assertFalse(g.complete({'latency':None,'error':None}))
        self.assertFalse(g.complete({'latency':float('nan'),'error':None}))
    def test_exact_unicode_gzip_restore(self):
        row,payload,_=self.artifact()
        actual,size=g.independent_restore(row,self.output)
        self.assertEqual(actual,payload)
        self.assertGreater(size,0)
    def test_audit_owner_rejected(self):
        row,_,_=self.artifact()
        row['question_id']='q2'
        with self.assertRaisesRegex(ValueError,'owner'):g.independent_restore(row,self.output)
    def test_gzip_crc_damage_rejected(self):
        row,_,path=self.artifact()
        raw=bytearray(path.read_bytes());raw[-8]^=1;path.write_bytes(raw)
        with self.assertRaises((gzip.BadGzipFile,OSError,EOFError)):g.independent_restore(row,self.output)
    def test_payload_sha_damage_rejected_even_valid_gzip(self):
        row,_,path=self.artifact()
        path.write_bytes(gzip.compress(b'{}',mtime=0))
        with self.assertRaisesRegex(ValueError,'SHA'):g.independent_restore(row,self.output)
    def test_allowlist_excludes_other_files_and_archive_is_exclusive(self):
        row,_,path=self.artifact()
        for name in g.BASE_FILES:(self.output/name).write_text('synthetic fixture')
        (self.output/'config.yml').write_text('synthetic excluded fixture')
        (self.output/'cache.pkl').write_text('synthetic excluded fixture')
        jp=self.journal(self.events())
        allowed=g.allowed_paths(self.output,{row['audit_artifacts']['path']},{jp.relative_to(self.output).as_posix()})
        self.assertEqual(len(allowed),9)
        self.assertNotIn('config.yml',allowed)
        self.assertNotIn('cache.pkl',allowed)
        files={n:{'sha256':g.sha(self.output/n),'bytes':(self.output/n).stat().st_size} for n in allowed}
        archive=self.root/'archive.zip'
        g.write_archive(archive,self.output,allowed,files)
        with zipfile.ZipFile(archive) as z:self.assertEqual(sorted(z.namelist()),allowed)
        original=archive.read_bytes()
        with self.assertRaises(FileExistsError):g.write_archive(archive,self.output,allowed,files)
        self.assertEqual(archive.read_bytes(),original)
    def test_symlink_owned_file_rejected(self):
        external=self.root/'external';external.write_text('synthetic excluded fixture')
        (self.output/'run_single.log').symlink_to(external)
        with self.assertRaisesRegex(ValueError,'unsafe'):g.owned_file(self.output,'run_single.log')
    def test_sdk_error_and_unknown_usage_are_retained(self):
        start,end=self.events(error=True)
        p=self.journal([start,end])
        result=g.parse_journal(p,self.output)
        self.assertTrue(result['pairing_complete'])
        self.assertTrue(result['safe_to_archive'])
        self.assertEqual(result['counts']['errors'],1)
        self.assertEqual(result['counts']['unknown_usage_ends'],1)
        self.assertEqual(result['cause_flags'],{'read':1,'timeout':1})
        self.assertEqual(result['http_status_counts'],{'None':1})
    def test_unmatched_start_not_credited_success(self):
        start,_=self.events()
        result=g.parse_journal(self.journal([start]),self.output)
        self.assertFalse(result['pairing_complete'])
        self.assertTrue(result['safe_to_archive'])
        self.assertEqual(result['counts']['unmatched_starts'],1)
        self.assertEqual(result['counts'].get('successes',0),0)
    def test_journal_private_fields_rejected(self):
        start,end=self.events();end['messages']=['synthetic poison']
        with self.assertRaisesRegex(ValueError,'unallowlisted'):g.parse_journal(self.journal([start,end]),self.output)
    def test_unvalidated_partial_tail_blocks_archive_not_invented_outcome(self):
        start,end=self.events()
        result=g.parse_journal(self.journal([start,end],b'{"event":'),self.output)
        self.assertFalse(result['safe_to_archive'])
        self.assertFalse(result['pairing_complete'])
        self.assertEqual(result['counts']['successes'],1)
        self.assertGreater(result['counts']['unvalidated_partial_bytes'],0)
    def test_journal_filename_must_own_events(self):
        start,end=self.events();start['pid']=124
        with self.assertRaisesRegex(ValueError,'owned'):g.parse_journal(self.journal([start,end]),self.output)
    def test_forbidden_source_config_path_rejected_before_read(self):
        for name in ('config.yml','.local-setup/config.backup.yml','../config.yml','/tmp/config.yml'):
            with self.assertRaises(ValueError):g.public_source_path(self.root,name)

    def test_exact_finite_json_distinguishes_boolean_numeric_and_float(self):
        self.assertFalse(g.json_equal(False,0))
        self.assertFalse(g.json_equal(1,True))
        self.assertFalse(g.json_equal({'answer':[False,1]}, {'answer':[0,True]}))
        self.assertFalse(g.json_equal(1,1.0))
        self.assertTrue(g.json_equal({'a':1,'b':[None]}, {'b':[None],'a':1}))
    def test_exact_json_rejects_nonfinite_values(self):
        for value in (float('nan'),float('inf'),float('-inf')):
            with self.assertRaises(ValueError):g.json_equal({'answer':value},{'answer':value})
    def test_report_collision_rejected_before_source_or_raw_reads(self):
        protected=self.root/'run_agent.py';protected.write_text('synthetic protected source')
        original=protected.read_bytes()
        with patch.object(g,'frozen_provenance') as provenance:
            with self.assertRaisesRegex(ValueError,'already exists'):
                g.main(['--root',str(self.root),'--owned_jobs_finished','--report',str(protected)])
            provenance.assert_not_called()
        self.assertEqual(protected.read_bytes(),original)
    def test_destinations_cannot_write_owned_output_or_each_other(self):
        record=self.root/'benchmarks/design-v4-20261014'
        with self.assertRaisesRegex(ValueError,'owned raw'):
            g.guard_destinations(self.root,record,self.output,self.output/'new.json')
        with self.assertRaisesRegex(ValueError,'protected repository'):
            g.guard_destinations(self.root,record,self.output,self.root/'agent/new.json')
        external=self.root.parent/(self.root.name+'-report.json')
        with self.assertRaisesRegex(ValueError,'collide'):
            g.guard_destinations(self.root,record,self.output,external,external/'raw.zip')
        g.guard_destinations(self.root,record,self.root/'output/experiments/current',record/'completion.json',self.root/'output/archives/fresh.zip')

    def v4_row(self,plan=True,error=None,planner_only=False,answer=1):
        import hashlib
        advisory={'status':'READY','steps':[{'id':'p1','quote':'weights','span':[0,7]}],'unknowns':[]} if plan else None
        phases=[{'phase':'planner','token_cap':10000,'tool_cap':0,'tokens':10,'tool_calls':0,'errors':[],'fatal':planner_only}]
        if not planner_only:phases.append({'phase':'solver','token_cap':190000,'tool_cap':30,'tokens':20,'tool_calls':3,'errors':[],'fatal':bool(error)})
        revision={'protocol':'plan-conditioned-v4','phases':phases,'plan':advisory,'plan_visible':plan,
                  'plan_sha256':hashlib.sha256(json.dumps(advisory,sort_keys=True,default=str,allow_nan=False).encode()).hexdigest() if plan else None,
                  'release_policy':'aggregate_accounting_failed' if planner_only else 'primary_failed_execution_or_source_contract' if error else 'source_valid_primary_preserved',
                  'final_check_issues':['failed'] if error else [],'solver_source_revised':False,'targeted_revised':False}
        trace=[]
        if plan and not planner_only:trace=[{'role':'system','content':'Frozen advisory plan JSON:\n'+json.dumps(advisory)}]
        row={'question_id':'q1','error':error,'agent_answer':answer,'revision':revision,'tokens':{'llm_calls':1},'budget':{'tool_calls_run':0}}
        return row,{**row,'trace':trace}
    def test_v4_plan_hash_and_actual_solver_visibility(self):
        row,restored=self.v4_row()
        counts=g.v4_protocol_counts(row,restored)
        self.assertEqual(counts['plan_validated_and_hashed'],1)
        self.assertEqual(counts['plan_observed_in_solver_system'],1)
        restored['revision']['plan_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'fingerprint'):g.v4_protocol_counts(row,restored)
    def test_v4_ignored_plan_has_no_model_advice(self):
        row,restored=self.v4_row(plan=False)
        counts=g.v4_protocol_counts(row,restored)
        self.assertEqual(counts['plan_ignored'],1)
        self.assertEqual(counts['plan_observed_in_solver_system'],0)
    def test_v4_planner_fatal_can_retain_plan_without_solver_exposure(self):
        row,restored=self.v4_row(error='unknown provider usage',planner_only=True,answer=None)
        counts=g.v4_protocol_counts(row,restored)
        self.assertEqual(counts['fatal_before_solver'],1)
        self.assertEqual(counts['declared_visible_but_fatal_before_solver'],1)
        self.assertEqual(counts['plan_observed_in_solver_system'],0)
    def test_v4_terminal_missing_protocol_is_recording_gap(self):
        row={'error':'uncaptured runtime failure','agent_answer':None}
        self.assertEqual(g.v4_protocol_counts(row,{'trace':[]})['terminal_errors_without_protocol_metadata'],1)
        row['error']=None
        with self.assertRaisesRegex(ValueError,'Nonterminal'):g.v4_protocol_counts(row,{'trace':[]})
    def test_v4_reported_invocation_and_host_overruns_are_retained(self):
        row,restored=self.v4_row(error='terminal accounting failure',answer=None)
        restored['trace'] += [{'role':'user','content':'Solver allocation remaining: synthetic'}]*21
        restored['trace'] += [{'role':'assistant','phase':'solver_boundary','model_call':False}]*2
        counts=g.v4_protocol_counts(row,restored)
        self.assertEqual(counts['reported_solver_invocation_limit_violations'],1)
        self.assertEqual(counts['reported_host_slot_limit_violations'],1)
    def test_v4_source_valid_null_is_separate_from_failed_null(self):
        row,restored=self.v4_row(answer=None)
        good=g.v4_protocol_counts(row,restored)
        self.assertEqual(good['source_valid_null_answers'],1)
        self.assertEqual(good['null_answers_outside_source_valid_release'],0)
        row,restored=self.v4_row(error='source gate failed',answer=None)
        bad=g.v4_protocol_counts(row,restored)
        self.assertEqual(bad['source_valid_null_answers'],0)
        self.assertEqual(bad['null_answers_outside_source_valid_release'],1)
    def test_v4_actual_advice_must_match_retained_plan(self):
        row,restored=self.v4_row()
        restored['trace'][0]['content']='Frozen advisory plan JSON:\n'+json.dumps({'status':'READY','steps':[],'unknowns':[]})
        with self.assertRaisesRegex(ValueError,'advisory context'):g.v4_protocol_counts(row,restored)

    def prospective(self):
        return {'model':'gpt-6-luna','fixed_questions_per_arm':300,'workers_per_arm':1,'first_attempt_only':True,
         'phase_caps':{'planner':[10000,0],'solver':[190000,30]},'solver_requested_tools':28,
         'reserved_counted_source_checks':2,'max_real_solver_calls':20,'max_host_slots':22,
         'global_caps':[200000,30],'estimated_input_guard':32000,'minimum_team_correct':239,
         'expected_public_binding':{'public':'synthetic'}}
    def test_v4_prospective_caps_are_checked(self):
        p=self.prospective()
        g.verify_prospective_protocol(p,{'public':'synthetic'})
        p['phase_caps']['solver'][1]=27
        with self.assertRaisesRegex(ValueError,'Prospective'):g.verify_prospective_protocol(p,{'public':'synthetic'})
    def test_v4_prospective_boolean_does_not_accept_numeric_one(self):
        p=self.prospective();p['first_attempt_only']=1
        with self.assertRaisesRegex(ValueError,'Prospective'):g.verify_prospective_protocol(p,{'public':'synthetic'})

if __name__=='__main__':unittest.main(verbosity=2)
