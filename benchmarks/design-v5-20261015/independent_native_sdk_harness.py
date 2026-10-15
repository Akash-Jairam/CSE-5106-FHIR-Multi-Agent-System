from copy import deepcopy
import json
import hashlib
from pathlib import Path
import tempfile
from unittest.mock import patch
import litellm
from openai import AzureOpenAI
import httpx2
import run_agent as runner
from scripts.audit_store import restore_result_audit
from tests import test_query_contract_agent as q
from tests.test_executable_critic_agent import first, proposal
from agent.query_contract_agent import HostBoundaryMessage
from utils import core_utils
source=Path('/private/tmp/fhir-v5-critic-implementation-1uhqxv2n/agent/executable_critic_agent.py')
source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()
assert source_sha256=='e66f7b3af92a63fd80c92f00a4bae357dc01cdf5b1d11c2f5d3ef51f3ddb42c7'
fixture=q.QueryContractIntegrationTests();fixture.setUp()
requests=[]
wrong=q.CODE.replace("'value': r['valueQuantity']['value']", "'value': 999")
responses=iter([q.fetch(),q.final_call(wrong,'wrong-citation'),q.final_call(q.CODE,'corrected-citation'),q.reply(json.dumps(first())),q.reply(json.dumps(proposal()))])
client=AzureOpenAI(azure_endpoint='https://offline-fixture.invalid',api_version='2024-12-01-preview',api_key='non-secret-unit-test-placeholder',timeout=60.0,max_retries=0)
assert isinstance(client._client,httpx2.Client)
assert isinstance(client._client._transport,httpx2.HTTPTransport)
def send(request,**kwargs):
    body=json.loads(request.content);requests.append(body)
    msg=q.FixtureMessage(**vars(next(responses))).model_dump()
    return httpx2.Response(200,request=request,json={'id':'offline-v5-review','object':'chat.completion','created':0,'model':'gpt-6-luna','choices':[{'index':0,'message':msg,'finish_reason':'tool_calls' if msg.get('tool_calls') else 'stop'}],'usage':{'prompt_tokens':100,'completion_tokens':50,'total_tokens':150}})
def completion(**kwargs):
    params={'model':'offline-deployment','messages':kwargs['messages'],'max_completion_tokens':min(kwargs.get('max_completion_tokens',16384),16384),'reasoning_effort':'none'}
    if kwargs.get('tools'):params['tools']=kwargs['tools']
    response=client.chat.completions.create(**params)
    return litellm.ModelResponse(**response.model_dump(exclude_none=True))
try:
    with tempfile.TemporaryDirectory() as directory,patch.object(client._client,'send',side_effect=send),patch.object(core_utils.litellm,'completion',side_effect=completion):
        job=runner.prepare_multiprocessing_inputs([q.QUESTION],'multi_agent_v5','gpt-6-luna',False,None,False,{'max_total_tokens':200000,'max_tool_calls':30},include_full_audit=True,audit_contexts=[{'output_dir':directory,'question_id':'synthetic-v5-review'}])[0]
        compact=runner.dispatch_agent_job(job)
        json.dumps(compact,allow_nan=False)
        restored=restore_result_audit(compact,directory,question_id='synthetic-v5-review')
        json.dumps(restored,allow_nan=False)
        assert (Path(directory)/compact['audit_artifacts']['path']).is_file()
        assert restored['error'] is None,restored['error']
        assert restored['agent_answer']==1
        assert restored['tokens']['llm_calls']==5
        assert restored['budget']['tool_calls_run']==8
        assert restored['revision']['primary_revision']['revised']
        assert restored['revision']['critic_revised']
        assert restored['revision']['final_check_issues']==[]
        assert len(requests)==5
        assert all(type(m) is dict for req in requests for m in req['messages'])
        assert [m for m in requests[2]['messages'] if m.get('content')==HostBoundaryMessage.content]==[HostBoundaryMessage().model_dump()]
        assert requests[3].get('tools') is None and requests[4].get('tools') is None
        assert requests[3]['max_completion_tokens']<=1800 and requests[4]['max_completion_tokens']<=4000
        assert hashlib.sha256(source.read_bytes()).hexdigest()==source_sha256
        report={'agent_sha256':source_sha256,'status':'PASS','scope':'model-free native default HTTPX2 SDK send mocked; no sockets/provider/configuration','sdk_calls':5,'tools':8,'critic_tools':restored['revision']['critic']['tool_calls'],'critic_calls':restored['revision']['critic']['model_calls'],'critic_completion_caps':[requests[3]['max_completion_tokens'],requests[4]['max_completion_tokens']],'release_policy':restored['revision']['release_policy'],'source_feedback':True,'gzip_roundtrip':True,'default_client':type(client._client).__module__+'.'+type(client._client).__name__,'default_transport':type(client._client._transport).__module__+'.'+type(client._client._transport).__name__}
        Path('/private/tmp/fhir-v5-independent-native-sdk-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
finally:
    client.close();fixture.doCleanups()
