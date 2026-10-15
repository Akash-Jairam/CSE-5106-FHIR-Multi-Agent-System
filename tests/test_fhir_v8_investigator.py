"""Independent synthetic V8 specifications; no benchmark data or paid inference.

Expected answers are handwritten.  Scripted native messages test execution,
lineage, fallback and accounting, not a model's ability to discover a repair.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from litellm import Message as ModelMessage
from tools import get_tool as real_tool

PATIENT = "11111111-1111-1111-1111-111111111111"
REFERENCE_TIME = "2131-01-01T00:00:00Z"


@dataclass(frozen=True)
class SemanticCase:
    name: str
    question: str
    resources: dict
    wrong_code: str
    correct_code: str
    expected_wrong: object
    expected_correct: object
    obligation: str
    probe_code: str

    @property
    def input_data(self):
        return (f"Question: {self.question}\nContext:\nPatient FHIR ID is {PATIENT}.\n"
                f"Assume the current time is {REFERENCE_TIME.replace('T', ' ').removesuffix('Z')}.")


def observation(identifier, value=None, *, day=1, label=None, unit="mg", **extra):
    row = {"resourceType": "Observation", "id": identifier,
           "subject": {"reference": "Patient/" + PATIENT},
           "effectiveDateTime": f"2130-01-{day:02d}T12:00:00Z",
           "code": {"text": label or identifier}, **extra}
    if value is not None:
        row["valueQuantity"] = {"value": value, "unit": unit}
    return row


def final_program(expression, *, prelude="", fields=("valueQuantity.value",), operation="Compute from the supplied records"):
    # Original field values are cited even when the requested answer is derived.
    return f"""{prelude}
rows = retrieved_resources['Observation']
answer = {expression}
evidence = []
for row in rows:
    for field in {list(fields)!r}:
        current = row
        for part in field.split('.'):
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if current is not None:
            evidence.append({{'resource_id': 'Observation/' + row['id'], 'field_path': field, 'value': current}})
contributing_ids = sorted({{item['resource_id'] for item in evidence}})
operation = {operation!r}
limitation = None
"""


def semantic_cases():
    tied = [observation("a", 2, label="Alpha"), observation("b", 9, label="Beta"),
            observation("c", 5, label="Gamma"), observation("d", 9, label="Delta")]
    extrema = "max(row['valueQuantity']['value'] for row in rows)"
    yield SemanticCase("tied_extrema", "Which names belong to all records with the greatest numeric value?",
        {"Observation": tied},
        final_program("max(rows, key=lambda row: row['valueQuantity']['value'])['code']['text']", fields=("valueQuantity.value", "code.text")),
        final_program("sorted(row['code']['text'] for row in rows if row['valueQuantity']['value'] == " + extrema + ")", fields=("valueQuantity.value", "code.text")),
        "Beta", ["Beta", "Delta"], "all records with the greatest numeric value",
        "rows=retrieved_resources['Observation']\nm=max(r['valueQuantity']['value'] for r in rows)\nanswer={'maximum':m,'tied_ids':sorted('Observation/'+r['id'] for r in rows if r['valueQuantity']['value']==m)}")
    projection = [observation("lo", 3, day=2), observation("hi", 8, day=3)]
    yield SemanticCase("requested_projection", "What timestamp belongs to the record with the lowest numeric value?",
        {"Observation": projection},
        final_program("min(rows, key=lambda row: row['valueQuantity']['value'])['valueQuantity']['value']", fields=("valueQuantity.value", "effectiveDateTime")),
        final_program("min(rows, key=lambda row: row['valueQuantity']['value'])['effectiveDateTime']", fields=("valueQuantity.value", "effectiveDateTime")),
        3, "2130-01-02T12:00:00Z", "What timestamp",
        "rows=retrieved_resources['Observation']\nr=min(rows,key=lambda x:x['valueQuantity']['value'])\nanswer={'minimum_value':r['valueQuantity']['value'],'requested_timestamp':r['effectiveDateTime'],'resource_id':'Observation/'+r['id']}")
    grouped = [observation("d1a", 4, day=1), observation("d1b", 10, day=1),
               observation("d2a", 3, day=2), observation("d2b", 12, day=2)]
    yield SemanticCase("group_then_aggregate", "What is the sum of the maximum numeric value for each calendar day?",
        {"Observation": grouped}, final_program(extrema),
        final_program("sum(max(r['valueQuantity']['value'] for r in rows if r['effectiveDateTime'][:10]==day) for day in sorted({r['effectiveDateTime'][:10] for r in rows}))", fields=("valueQuantity.value", "effectiveDateTime")),
        12, 22, "for each calendar day",
        "rows=retrieved_resources['Observation']\nanswer={day:max(r['valueQuantity']['value'] for r in rows if r['effectiveDateTime'][:10]==day) for day in sorted({r['effectiveDateTime'][:10] for r in rows})}")
    bounds = [observation("lower", 1, day=1), observation("middle", 2, day=2),
              observation("upper", 3, day=3), observation("outside", 8, day=4)]
    yield SemanticCase("strict_inclusive_time_bounds", "Sum values strictly after 2130-01-01 and through 2130-01-03, comparing calendar dates.",
        {"Observation": bounds},
        final_program("sum(r['valueQuantity']['value'] for r in rows if '2130-01-01' <= r['effectiveDateTime'][:10] < '2130-01-03')", fields=("valueQuantity.value", "effectiveDateTime")),
        final_program("sum(r['valueQuantity']['value'] for r in rows if '2130-01-01' < r['effectiveDateTime'][:10] <= '2130-01-03')", fields=("valueQuantity.value", "effectiveDateTime")),
        3, 5, "strictly after 2130-01-01 and through 2130-01-03",
        "rows=retrieved_resources['Observation']\nanswer=[{'id':'Observation/'+r['id'],'date':r['effectiveDateTime'][:10],'value':r['valueQuantity']['value']} for r in rows]")
    relative = [observation("early", 1, day=1), observation("yesterday", 2, day=2), observation("today", 3, day=3)]
    yield SemanticCase("relative_day_window", "As of 2130-01-03, how many supplied records are from yesterday or today?",
        {"Observation": relative}, final_program("len(rows)"),
        final_program("sum(r['effectiveDateTime'][:10] in ('2130-01-02','2130-01-03') for r in rows)", fields=("effectiveDateTime",)),
        3, 2, "yesterday or today",
        "answer=sorted(r['effectiveDateTime'][:10] for r in retrieved_resources['Observation'])")
    units = [observation("milligrams", 500, unit="mg"), observation("grams", 1.5, unit="g")]
    yield SemanticCase("unit_conversion_before_sum", "What is the total mass in grams across the supplied records?",
        {"Observation": units}, final_program("sum(r['valueQuantity']['value'] for r in rows)", fields=("valueQuantity.value", "valueQuantity.unit")),
        final_program("sum(r['valueQuantity']['value'] / 1000 if r['valueQuantity']['unit']=='mg' else r['valueQuantity']['value'] for r in rows)", fields=("valueQuantity.value", "valueQuantity.unit")),
        501.5, 2.0, "in grams",
        "answer=[{'id':'Observation/'+r['id'],'quantity':r['valueQuantity']} for r in retrieved_resources['Observation']]")
    linked = [observation("group", label="requested group", hasMember=[{"reference":"Observation/linked"}]),
              observation("linked", 7), observation("unrelated", 100)]
    yield SemanticCase("follow_only_declared_links", "What is the numeric total of members linked by the requested group record?",
        {"Observation": linked}, final_program("sum(r.get('valueQuantity',{}).get('value',0) for r in rows)"),
        final_program("sum(r.get('valueQuantity',{}).get('value',0) for r in rows if 'Observation/'+r['id'] in {m['reference'] for parent in rows if parent['code']['text']=='requested group' for m in parent.get('hasMember',[])})", fields=("valueQuantity.value", "hasMember")),
        107, 7, "members linked by the requested group record",
        "answer=[{'id':'Observation/'+r['id'],'links':r.get('hasMember',[]),'value':r.get('valueQuantity')} for r in retrieved_resources['Observation']]")


class NativeMessage(ModelMessage):
    def __init__(self, content=None, calls=None):
        super().__init__(role="assistant", content=content, tool_calls=calls)


def native_tool(code, *, mode="final", identifier="synthetic-native", name="execute_python_code", **arguments):
    arguments = {"code":code, "mode":mode, **arguments} if name == "execute_python_code" else arguments
    return NativeMessage(calls=[{"id":identifier,"type":"function","function":{
        "name":name,"arguments":json.dumps(arguments)}}])


def known_usage(prompt=100, completion=80, *, known=True):
    return {"prompt_tokens":prompt,"completion_tokens":completion,"total_tokens":prompt+completion,
        "llm_calls":1,"cost":0.0 if known else None,"per_call":[{
            "prompt":prompt,"completion":completion,"cached":0,"usage_available":known}]}


def assert_native_message_protocol(messages):
    """Independent send-boundary spec: finish every native block before prose.

    This deliberately does not reuse agent message helpers or witness logic.
    Matching replies may be in any order, but every call needs exactly one reply.
    """
    pending = set()
    for index, raw in enumerate(messages):
        message = raw if isinstance(raw, dict) else raw.model_dump()
        role = message.get('role')
        if pending and role != 'tool':
            raise AssertionError(f'message {index}: {role} before replies for {sorted(pending)}')
        if role == 'tool':
            identifier = message.get('tool_call_id')
            if identifier not in pending:
                raise AssertionError(f'message {index}: orphan or duplicate tool reply {identifier!r}')
            pending.remove(identifier)
            continue
        calls = message.get('tool_calls') or []
        if calls and role != 'assistant':
            raise AssertionError(f'message {index}: native calls outside assistant')
        for raw_call in calls:
            call = raw_call if isinstance(raw_call, dict) else raw_call.model_dump()
            identifier = call.get('id')
            if not isinstance(identifier, str) or not identifier or identifier in pending:
                raise AssertionError(f'message {index}: missing or duplicate native call ID')
            pending.add(identifier)
    if pending:
        raise AssertionError(f'missing native replies for {sorted(pending)}')


class NativeMessageProtocolTests(unittest.TestCase):
    def test_walker_accepts_complete_multi_call_block_before_feedback(self):
        calls = [native_tool('answer=1', identifier=identifier).tool_calls[0]
                 for identifier in ('first', 'second')]
        assert_native_message_protocol([NativeMessage('witness', calls=calls),
            {'role':'tool', 'tool_call_id':'second', 'content':'known failure'},
            {'role':'tool', 'tool_call_id':'first', 'content':'success'},
            {'role':'user', 'content':'Witness lineage recorded'}])

    def test_walker_rejects_interleaving_missing_or_duplicate_native_replies(self):
        assistant = native_tool('answer=1', identifier='pending')
        reply = {'role':'tool', 'tool_call_id':'pending', 'content':'success'}
        other = native_tool('answer=2', identifier='other')
        bad = [[assistant, {'role':'user', 'content':'early witness feedback'}, reply],
               [assistant, NativeMessage('early assistant feedback'), reply],
               [assistant], [reply], [assistant, reply, deepcopy(reply)],
               [assistant, {'role':'tool', 'tool_call_id':'unmatched', 'content':'bad'}],
               [NativeMessage(calls=[*assistant.tool_calls, *other.tool_calls]), reply],
               [NativeMessage(calls=[*assistant.tool_calls, *assistant.tool_calls])]]
        for messages in bad:
            with self.subTest(messages=messages), self.assertRaises(AssertionError):
                assert_native_message_protocol(messages)


class InvestigatorSpecificationFixtures(unittest.TestCase):
    def test_handwritten_expectations_use_actual_isolated_worker(self):
        execute = real_tool("execute_python_code")
        for case in semantic_cases():
            with self.subTest(case=case.name):
                wrong = execute(case.wrong_code, mode="final", global_vars={"retrieved_resources":deepcopy(case.resources)})
                correct = execute(case.correct_code, mode="final", global_vars={"retrieved_resources":deepcopy(case.resources)})
                self.assertNotIn("error",wrong,wrong)
                self.assertNotIn("error",correct,correct)
                self.assertEqual(wrong["answer"],case.expected_wrong)
                self.assertEqual(correct["answer"],case.expected_correct)
                self.assertNotEqual(wrong["answer"],correct["answer"])
                self.assertIn(case.obligation,case.question)

    def test_tied_extrema_correct_program_is_order_invariant(self):
        case=next(semantic_cases()); execute=real_tool("execute_python_code")
        original=case.resources["Observation"]
        for rows in (original,list(reversed(original)),original[1:]+original[:1]):
            with self.subTest(order=[r["id"] for r in rows]):
                result=execute(case.correct_code,mode="final",global_vars={"retrieved_resources":{"Observation":deepcopy(rows)}})
                self.assertEqual(result["answer"],["Beta","Delta"])


class StudyCacheIndependentTests(unittest.TestCase):
    def test_each_followup_starts_from_same_post_A_bytes_without_shared_inodes(self):
        from scripts.v8_study_cache import CaseCache
        with tempfile.TemporaryDirectory() as tmp:
            cache=CaseCache(Path(tmp)/"case")
            self.assertEqual(list(cache.primary.iterdir()),[])
            (cache.primary/"A-held.cache").write_bytes(b"synthetic-cache-byte-envelope")
            proof=cache.freeze()
            arms={mode:cache.fork(mode) for mode in ("R","W","C")}
            self.assertEqual(proof["files"],1)
            self.assertEqual(len({(directory/"A-held.cache").stat().st_ino for directory in arms.values()}),3)
            self.assertNotIn((cache.base/"A-held.cache").stat().st_ino,
                             {(directory/"A-held.cache").stat().st_ino for directory in arms.values()})
            for directory in arms.values():
                self.assertEqual((directory/"A-held.cache").read_bytes(),b"synthetic-cache-byte-envelope")
            # R may warm/replace its own cache; neither W/C nor immutable A changes.
            (arms["R"]/"R-only.cache").write_bytes(b"new-source-response")
            (arms["R"]/"A-held.cache").write_bytes(b"R-private-replacement")
            for mode in ("W","C"):
                self.assertEqual((arms[mode]/"A-held.cache").read_bytes(),b"synthetic-cache-byte-envelope")
                self.assertFalse((arms[mode]/"R-only.cache").exists())
            self.assertEqual(cache.after_followup("R")["new_overlay_files"],1)
            self.assertTrue(cache.after_followup("C")["base_unchanged"])
            self.assertEqual(cache.proof(),proof)

    def test_immutable_A_cache_mutation_is_detected_before_next_arm(self):
        from scripts.v8_study_cache import CaseCache
        with tempfile.TemporaryDirectory() as tmp:
            cache=CaseCache(Path(tmp)/"case")
            (cache.primary/"initial.cache").write_bytes(b"one")
            cache.freeze(); cache.fork("W")
            (cache.base/"initial.cache").write_bytes(b"two")
            with self.assertRaisesRegex(ValueError,"base changed"):
                cache.fork("R")

    def test_cache_cannot_fork_twice_or_accept_an_unplanned_arm(self):
        from scripts.v8_study_cache import CaseCache
        with tempfile.TemporaryDirectory() as tmp:
            cache=CaseCache(Path(tmp)/"case"); cache.freeze(); cache.fork("W")
            for mode in ("W","oracle"):
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    cache.fork(mode)

    def test_cache_hashing_never_unpickles_and_rejects_linked_entries(self):
        from scripts.v8_study_cache import CaseCache
        with tempfile.TemporaryDirectory() as tmp:
            cache=CaseCache(Path(tmp)/"case")
            # Deliberately not a valid pickle: inventories must inspect bytes only.
            (cache.primary/"opaque.cache").write_bytes(b"not-a-pickle")
            cache.freeze(); self.assertEqual(cache.proof()["files"],1)
            other=CaseCache(Path(tmp)/"other")
            (other.primary/"linked.cache").symlink_to(cache.base/"opaque.cache")
            with self.assertRaises(ValueError):other.freeze()

    def test_global_cache_context_is_sequential_and_restored_on_error(self):
        from scripts.v8_study_cache import activated_cache
        from types import SimpleNamespace
        module=SimpleNamespace(CACHE_DIR="prior-private-dir",CACHE_ENABLED=False)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,"synthetic tool failure"):
                with activated_cache(tmp,module):
                    self.assertEqual(module.CACHE_DIR,str(Path(tmp).resolve()))
                    self.assertTrue(module.CACHE_ENABLED)
                    with self.assertRaisesRegex(RuntimeError,"sequential"):
                        with activated_cache(tmp,module):pass
                    raise RuntimeError("synthetic tool failure")
            self.assertEqual((module.CACHE_DIR,module.CACHE_ENABLED),("prior-private-dir",False))
            with activated_cache(tmp,module):self.assertTrue(module.CACHE_ENABLED)
            self.assertEqual((module.CACHE_DIR,module.CACHE_ENABLED),("prior-private-dir",False))


def counterexample_claim(case, *, observation_id="observation_1", **changes):
    return {"obligation_span":case.obligation,"observation_id":observation_id,
            "A_behavior":"The actual A program returns " + repr(case.expected_wrong),
            "contradiction":"The retained probe exposes a different requested projection or operand set.", **changes}


def with_claim(message, claim):
    message.content=json.dumps(claim)
    return message


def adjudication(case, *, choice="B", a_status="contradicted", b_status="supported",
                 observation_id="observation_1", span=None):
    return NativeMessage(json.dumps({"choice":choice,"reason":"Compare the actual complete computations with the executed observation.",
        "obligation_span":case.obligation if span is None else span,"observation_id":observation_id,
        "requirements":[{"requirement":case.obligation,"A":{"status":a_status,"basis":"A's actual computation conflicts with this observed operand or projection."},
            "B":{"status":b_status,"basis":"B executes a complete computation using the actual cited records."}}]}))


class InvestigatorCoreIndependentTests(unittest.TestCase):
    def exercise(self, case, auxiliary, *, mode="R", primary_code=None, bills=None,
                 input_data=None, checker=None, envelope_change=None, retrieval=None,
                 primary_replies=None, real_tokenizer=False, handler_error_ids=()):
        from agent.counterexample_investigator_agent import CounterexampleInvestigatorAgent
        from agent.single_agent import SingleAgent
        from tools.field_check import check_fields
        requests=[]; executions=[]; source_reads=[]; fetches=[]; protocol_failures=[]
        primary=primary_replies if primary_replies is not None else [
            native_tool(primary_code or case.wrong_code,identifier="A-final"),
            NativeMessage('{"answer":999,"evidence":[],"limitation":null}')]
        responses=iter(primary+list(auxiliary)); charges=iter(bills) if bills is not None else None
        resources=deepcopy(case.resources)
        def source_fetch(refs):
            source_reads.append(deepcopy(refs))
            return {row['id']:deepcopy(row) for kind,rows in resources.items() for row in rows if (kind,row['id']) in refs}
        def dispatch(name):
            if name == "execute_python_code":
                execute=real_tool(name)
                def observed_execute(**kwargs):
                    executions.append(deepcopy(kwargs));return execute(**kwargs)
                return observed_execute
            if name in ("get_resources_by_patient_fhir_id","get_resources_by_resource_id"):
                def fetch(**kwargs):
                    fetches.append(deepcopy(kwargs))
                    if isinstance(retrieval,Exception):raise retrieval
                    return deepcopy(resources if retrieval is None else retrieval)
                return fetch
            if name == "fhir_request_get":
                def forbidden(**kwargs):
                    fetches.append({'unadvertised_tool_executed':deepcopy(kwargs)})
                    return {'error':'An unadvertised tool must never reach this spy'}
                return forbidden
            return real_tool(name)
        def provider(**kwargs):
            try:
                assert_native_message_protocol(kwargs['messages'])
            except AssertionError as exc:
                protocol_failures.append(str(exc))
                raise
            requests.append(deepcopy(kwargs))
            try:response=next(responses)
            except StopIteration as exc:raise AssertionError('Unexpected extra model request') from exc
            if charges is not None:
                charge=deepcopy(next(charges))
            elif real_tokenizer:
                from utils.core_utils import count_tokens_in_messages
                charge=known_usage(prompt=count_tokens_in_messages(kwargs['messages'],kwargs.get('tools')))
            else:
                charge=known_usage()
            return response,None,charge
        question=input_data or case.input_data
        with ExitStack() as stack:
            stack.enter_context(patch('agent.base_agent.setup_api_keys'))
            if not real_tokenizer:
                stack.enter_context(patch('agent.base_agent.count_tokens_in_messages',return_value=10))
            stack.enter_context(patch('agent.base_agent.safe_llm_call',side_effect=provider))
            stack.enter_context(patch('agent.single_agent.get_starting_packet',side_effect=lambda *_a,**_kw:{'text':'Complete synthetic held records','resources':deepcopy(case.resources)}))
            stack.enter_context(patch('tools.field_check.fetch_from_store',side_effect=source_fetch))
            stack.enter_context(patch('agent.multi_turn_code_resource_agent.get_tool',side_effect=dispatch))
            if handler_error_ids:
                original_handler=SingleAgent._handle_tool_call
                def handler(agent,call,workspace):
                    if call.id in handler_error_ids:
                        raise RuntimeError('Synthetic native handler failure')
                    return original_handler(agent,call,workspace)
                stack.enter_context(patch.object(SingleAgent,'_handle_tool_call',handler))
            if checker is not None:stack.enter_context(patch('agent.single_agent.check_fields',side_effect=checker))
            primary_agent=CounterexampleInvestigatorAgent('synthetic-v8',mode=mode)
            primary_agent.set_budget(max_total_tokens=200000,max_tool_calls=30)
            envelope=primary_agent.run_primary(question)
            before=deepcopy(envelope)
            if envelope_change is not None:envelope_change(envelope)
            followup=CounterexampleInvestigatorAgent('synthetic-v8',mode=mode)
            followup.set_budget(max_total_tokens=200000,max_tool_calls=30)
            result=followup.run_from_primary(question,envelope,mode=mode)
        self.assertEqual(protocol_failures,[], 'Invalid native protocol at mocked model-send boundary')
        assert_native_message_protocol(result['trace'])
        if envelope_change is None:self.assertEqual(envelope,before,'Auxiliary mutated the immutable A envelope')
        return result,requests,envelope,executions,source_reads,fetches

    def successful_script(self,case,*,candidate=None,claim=None,judge=None):
        return [native_tool(case.probe_code,mode='inspect',identifier='probe'),
                with_claim(native_tool(candidate or case.correct_code,identifier='B-final'),claim or counterexample_claim(case)),
                judge or adjudication(case)]

    def test_accepted_witness_with_probe_replies_before_next_investigator_send(self):
        case=next(semantic_cases())
        script=[native_tool(case.probe_code,mode='inspect',identifier='first-observation'),
                with_claim(native_tool(case.probe_code,mode='inspect',identifier='witness-probe'),
                           counterexample_claim(case)),
                native_tool(case.correct_code,identifier='B-after-witness-probe'),adjudication(case)]
        for mode in ('W','R','C'):
            with self.subTest(mode=mode):
                result,requests,_,executions,source_reads,_=self.exercise(case,script,mode=mode)
                self.assertIsNone(result['error'])
                self.assertEqual(result['answer'],case.expected_correct)
                self.assertEqual(len(requests),6)
                self.assertEqual(len(executions),4)
                self.assertEqual(len(source_reads),2)
                messages=requests[4]['messages']
                reply=next(i for i,m in enumerate(messages)
                    if isinstance(m,dict) and m.get('tool_call_id')=='witness-probe')
                self.assertEqual(messages[reply+1]['content'],
                    'Witness lineage recorded; this does not prove its semantics.')
                self.assertTrue(result['revision']['witness'])

    def test_rejected_witness_with_probe_replies_before_repaired_witness_send(self):
        case=next(semantic_cases())
        script=[native_tool(case.probe_code,mode='inspect',identifier='first-observation'),
                with_claim(native_tool(case.probe_code,mode='inspect',identifier='rejected-witness-probe'),
                           counterexample_claim(case,observation_id='observation_not_received')),
                with_claim(native_tool(case.correct_code,identifier='B-with-valid-witness'),
                           counterexample_claim(case)),adjudication(case)]
        result,requests,_,_,_,_=self.exercise(case,script)
        self.assertIsNone(result['error'])
        self.assertEqual(result['answer'],case.expected_correct)
        self.assertEqual(len(requests),6)
        messages=requests[4]['messages']
        reply=next(i for i,m in enumerate(messages)
            if isinstance(m,dict) and m.get('tool_call_id')=='rejected-witness-probe')
        self.assertTrue(messages[reply+1]['content'].startswith('No witness admitted (ValueError).'))
        self.assertEqual(result['revision']['witness']['observation_id'],'observation_1')

    def test_multi_call_witness_feedback_waits_for_success_failure_and_refusal_replies(self):
        case=next(semantic_cases())
        for valid in (True,False):
            with self.subTest(witness_admitted=valid):
                calls=[native_tool(case.probe_code,mode='inspect',identifier='batch-probe').tool_calls[0],
                       native_tool(case.correct_code,identifier='handler-raises').tool_calls[0],
                       native_tool('',name='fhir_request_get',identifier='batch-refused').tool_calls[0]]
                claim=counterexample_claim(case,observation_id='observation_1' if valid else 'unreceived')
                batch=NativeMessage(json.dumps(claim),calls=calls)
                result,requests,_,executions,source_reads,fetches=self.exercise(case,
                    [native_tool(case.probe_code,mode='inspect',identifier='first-observation'),batch],
                    handler_error_ids=('handler-raises',))
                self.assertIsNone(result['error'])
                self.assertEqual(result['answer'],case.expected_wrong)
                self.assertEqual(len(requests),4)
                self.assertEqual(len(executions),3)  # A final plus two successful probes
                self.assertEqual(len(source_reads),1)  # no eligible B to source-check
                self.assertEqual(fetches,[])
                self.assertEqual(result['budget']['tool_calls_run'],5)  # refused call uncharged
                trace=result['trace']
                assert_native_message_protocol(trace)
                first=next(i for i,m in enumerate(trace) if m.get('tool_call_id')=='batch-probe')
                replies=trace[first:first+3]
                self.assertEqual([m['role'] for m in replies],['tool']*3)
                self.assertEqual([m['tool_call_id'] for m in replies],
                                 ['batch-probe','handler-raises','batch-refused'])
                self.assertIn('Native tool raised RuntimeError',replies[1]['content'])
                self.assertIn('Not run: V8 admission:',replies[2]['content'])
                feedback=trace[first+3]
                self.assertEqual(feedback['role'],'user')
                self.assertTrue(feedback['content'].startswith(
                    'Witness lineage recorded;' if valid else 'No witness admitted (ValueError).'))
                self.assertEqual(result['revision']['witness'] is not None,valid)

    def test_counterexamples_recompute_complete_candidates_through_actual_core(self):
        for case in semantic_cases():
            # Relative-date source convention is separately tested as an ineligible
            # primary; this fixture's date wording is intentionally not a checker oracle.
            if case.name == 'relative_day_window':continue
            with self.subTest(case=case.name):
                result,requests,envelope,executions,source_reads,_=self.exercise(case,self.successful_script(case))
                self.assertIsNone(result['error'])
                self.assertEqual(result['answer'],case.expected_correct)
                self.assertEqual(len(requests),5)  # natural A2 + probe/final/judge3
                self.assertEqual(len(executions),3)
                self.assertEqual(executions[1]['mode'],'inspect')
                self.assertEqual(executions[2]['mode'],'final')
                self.assertIsNone(requests[-1].get('tools'))
                self.assertGreaterEqual(len(source_reads),2)
                self.assertEqual(result['tokens']['llm_calls'],5)

    def test_correct_A_agreement_keeps_actual_release_without_judging(self):
        case=next(semantic_cases())
        script=self.successful_script(case,candidate=case.correct_code)[:2]
        result,requests,_,_,_,_=self.exercise(case,script,primary_code=case.correct_code)
        self.assertIsNone(result['error'])
        self.assertEqual(result['answer'],case.expected_correct)
        self.assertEqual(len(requests),4)

    def test_nonliteral_or_unobserved_counterexample_cannot_authorize_switch(self):
        case=next(semantic_cases())
        claims=[counterexample_claim(case,obligation_span='an invented question requirement'),
                counterexample_claim(case,observation_id='observation_999')]
        for claim in claims:
            with self.subTest(claim=claim):
                result,requests,_,_,_,_=self.exercise(case,self.successful_script(case,claim=claim))
                self.assertIsNone(result['error'])
                self.assertEqual(result['answer'],case.expected_wrong)
                self.assertLessEqual(len(requests),5)

    def test_judge_cannot_replace_the_validated_obligation_or_observation(self):
        case=next(semantic_cases())
        for judge in (adjudication(case,span='unbound obligation'),adjudication(case,observation_id='observation_999')):
            with self.subTest(judge=judge.content):
                result,requests,_,_,_,_=self.exercise(case,self.successful_script(case,judge=judge))
                self.assertIsNone(result['error'])
                self.assertEqual(result['answer'],case.expected_wrong)
                self.assertEqual(len(requests),5)

    def test_missing_A_contradiction_or_B_support_preserves_A(self):
        case=next(semantic_cases())
        for judge in (adjudication(case,a_status='not_found'),adjudication(case,b_status='not_found'),adjudication(case,choice='UNKNOWN')):
            with self.subTest(judge=judge.content):
                result,_,_,_,_,_=self.exercise(case,self.successful_script(case,judge=judge))
                self.assertIsNone(result['error'])
                self.assertEqual(result['answer'],case.expected_wrong)

    def test_unknown_probe_or_judge_usage_is_fatal_even_after_valid_A(self):
        case=next(semantic_cases())
        for position in (2,4):
            bills=[known_usage() for _ in range(5)];bills[position]=known_usage(known=False)
            with self.subTest(position=position):
                result,requests,envelope,_,_,_=self.exercise(case,self.successful_script(case),bills=bills)
                self.assertIsNotNone(result['error'])
                self.assertIsNone(result['answer'])
                self.assertEqual(result['tokens']['llm_calls'],position+1)
                self.assertFalse(result['tokens']['per_call'][-1]['usage_available'])
                self.assertEqual(len(requests),position+1)

    def test_phase_or_residual_token_overshoot_cannot_hide_behind_global_cap(self):
        case=next(semantic_cases())
        for a_bill,probe_bill in ((known_usage(),known_usage(prompt=60000,completion=1000)),
                                  (known_usage(prompt=74900,completion=100),known_usage(prompt=50000,completion=1000))):
            with self.subTest(a_tokens=a_bill['total_tokens']):
                result,requests,_,_,_,_=self.exercise(case,self.successful_script(case),bills=[a_bill,a_bill,probe_bill])
                self.assertIsNotNone(result['error'])
                self.assertIsNone(result['answer'])
                self.assertEqual(len(requests),3)

    def test_primary_keeps_all30_native_calls_and_leaves_no_extra_send(self):
        case=next(semantic_cases())
        primary=[NativeMessage('Continue examining the supplied records') for _ in range(28)]
        primary += [native_tool(case.wrong_code,identifier='A-last-final'),NativeMessage('{"answer":999,"evidence":[],"limitation":null}')]
        result,requests,envelope,executions,_,_=self.exercise(case,[],primary_replies=primary)
        self.assertIsNone(result['error'])
        self.assertEqual(result['answer'],case.expected_wrong)
        self.assertEqual(len(requests),30)
        self.assertEqual(result['tokens']['llm_calls'],30)
        self.assertEqual(len(executions),1)
        limits=envelope['payload']['candidate']['diagnostics']
        self.assertEqual((limits['token_cap'],limits['tool_cap'],limits['sdk_cap']),(200000,30,30))

    def test_prose_A_is_preserved_without_manufacturing_an_executable_candidate(self):
        case=next(semantic_cases())
        prose=NativeMessage(json.dumps({'answer':17,'evidence':[{'resource_id':'Observation/b',
            'field_path':'valueQuantity.value','value':9}],'limitation':'No executable candidate'}))
        result,requests,envelope,executions,_,_=self.exercise(case,[],primary_replies=[prose])
        self.assertIsNone(result['error'])
        self.assertEqual(result['answer'],17)
        self.assertEqual(len(requests),1)
        self.assertEqual(executions,[])
        self.assertIsNone(envelope['payload']['candidate']['artifact'])

    def test_computed_but_source_withheld_A_is_skipped_not_replayed_or_rescued(self):
        case=next(semantic_cases())
        def refused(citations,**_):
            return {'issues':['scope: synthetic unresolved original-source membership'],'n_valid':0,
                'results':[{'citation':deepcopy(c),'status':'invalid','warnings':[]} for c in citations]}
        primary=[native_tool(case.wrong_code,identifier='A-first'),NativeMessage('{"answer":999,"evidence":[],"limitation":null}'),
                 native_tool(case.wrong_code,identifier='A-source-feedback'),NativeMessage('{"answer":999,"evidence":[],"limitation":null}')]
        result,requests,envelope,executions,_,_=self.exercise(case,[],checker=refused,primary_replies=primary)
        self.assertIsNone(result['error'])
        self.assertIsNone(result['answer'])
        self.assertEqual(len(requests),4)
        self.assertEqual(len(executions),2)
        a=envelope['payload']['candidate']
        self.assertIsNone(a['artifact'])
        self.assertEqual(a['released']['revision']['final']['answer'],case.expected_wrong)

    def test_failed_probe_does_not_become_an_observation_proving_absence(self):
        case=next(semantic_cases())
        broken=native_tool("raise ValueError('synthetic query is unavailable')",mode='inspect',identifier='probe-fails')
        script=[broken,with_claim(native_tool(case.correct_code,identifier='B-after-failure'),counterexample_claim(case)),adjudication(case)]
        script += [NativeMessage('Unknown; no qualifying executed observation') for _ in range(6)]
        result,requests,_,_,_,_=self.exercise(case,script)
        self.assertIsNone(result['error'])
        self.assertEqual(result['answer'],case.expected_wrong)
        self.assertLessEqual(len(requests),8)

    def test_W_never_retrieves_and_R_tool_failure_never_proves_zero(self):
        case=next(semantic_cases())
        query=native_tool('',name='get_resources_by_patient_fhir_id',identifier='attempted-patient-query',
                          resource_type='Observation',patient_fhir_id=PATIENT)
        script=[query]+[NativeMessage('Unknown; unavailable retrieval does not establish absence') for _ in range(7)]
        result,requests,_,_,_,fetches=self.exercise(case,script,mode='W')
        self.assertIsNone(result['error'])
        self.assertEqual(result['answer'],case.expected_wrong)
        self.assertEqual(fetches,[])
        self.assertLessEqual(len(requests),8)
        result,requests,_,_,_,fetches=self.exercise(case,script,mode='R',retrieval=RuntimeError('synthetic store unavailable'))
        self.assertIsNone(result['error'])
        self.assertEqual(result['answer'],case.expected_wrong)
        self.assertEqual(len(fetches),1)
        self.assertLessEqual(len(requests),8)

    def test_six_auxiliary_sends_and_six_tools_can_finish_without_a_hidden_request(self):
        case=next(semantic_cases())
        invalid="raise ValueError('synthetic recoverable code failure')"
        script=[native_tool(case.probe_code,mode='inspect',identifier='probe-one'),
                native_tool(case.probe_code,mode='inspect',identifier='probe-two'),
                with_claim(native_tool(invalid,identifier='bad-program-one'),counterexample_claim(case,observation_id='observation_2')),
                with_claim(native_tool(invalid,identifier='bad-program-two'),counterexample_claim(case,observation_id='observation_2')),
                with_claim(native_tool(case.correct_code,identifier='last-program'),counterexample_claim(case,observation_id='observation_2')),
                adjudication(case,observation_id='observation_2')]
        result,requests,_,_,_,_=self.exercise(case,script)
        self.assertIsNone(result['error'])
        self.assertEqual(result['answer'],case.expected_correct)
        self.assertEqual(len(requests)-2,6)
        self.assertEqual(result['budget']['tool_calls_run']-2,6)
        self.assertEqual(result['tokens']['llm_calls'],len(requests))

    def test_correct_A_and_harmful_dependent_repair_require_fallible_judgment(self):
        rows=[observation('included-a',8,label='include',duration={'value':2}),
              observation('included-b',12,label='include',duration={'value':3}),
              observation('distractor',100,label='exclude',duration={'value':5})]
        total="sum(r['valueQuantity']['value'] for r in rows if r['code']['text']=='include')"
        scoped_duration="sum(r['duration']['value'] for r in rows if r['code']['text']=='include')"
        all_duration="sum(r['duration']['value'] for r in rows)"
        good=final_program(total+'/'+scoped_duration,fields=('valueQuantity.value','code.text','duration.value'))
        harmful=final_program(total+'/'+all_duration,fields=('valueQuantity.value','code.text','duration.value'))
        case=SemanticCase('dependent_rate','Compute total value divided by total duration for the included records only.',
            {'Observation':rows},harmful,good,2.0,4.0,'for the included records only',
            "answer=[{'id':'Observation/'+r['id'],'group':r['code']['text'],'duration':r['duration']['value']} for r in retrieved_resources['Observation']]")
        # The replacement changes selection but leaves a dependent denominator
        # wrong.  Source validity alone cannot establish semantic support.
        for choice,expected in [('A',4.0),('B',2.0)]:
            with self.subTest(judge=choice):
                script=self.successful_script(case,candidate=harmful,judge=adjudication(case,choice=choice))
                result,_,_,_,_,_=self.exercise(case,script,primary_code=good)
                self.assertIsNone(result['error'])
                self.assertEqual(result['answer'],expected)

    def test_seventh_send_path_is_known_fallback_without_spending_an_extra_request(self):
        case=next(semantic_cases())
        script=[native_tool(case.probe_code,mode='inspect',identifier='probe-one'),
                native_tool(case.probe_code,mode='inspect',identifier='probe-two')]
        script += [with_claim(native_tool("raise ValueError('recoverable')",identifier=f'bad-{i}'),
                    counterexample_claim(case,observation_id='observation_2')) for i in range(3)]
        script += [with_claim(native_tool(case.correct_code,identifier='cannot-fit-final'),
                    counterexample_claim(case,observation_id='observation_2')),
                   adjudication(case,observation_id='observation_2')]
        result,requests,_,_,_,_=self.exercise(case,script)
        self.assertIsNone(result['error'])
        self.assertEqual(result['answer'],case.expected_wrong)
        self.assertLessEqual(len(requests)-2,6)
        self.assertLessEqual(result['budget']['tool_calls_run']-2,6)
        self.assertEqual(result['tokens']['llm_calls'],len(requests))

    def test_unadvertised_registered_request_tool_never_executes_or_becomes_observation(self):
        case=next(semantic_cases())
        for mode in ('W','R'):
            with self.subTest(mode=mode):
                forbidden=native_tool('',name='fhir_request_get',identifier='not-advertised',
                    resource_type='Observation',patient_fhir_id=PATIENT)
                result,requests,_,_,_,fetches=self.exercise(case,[forbidden,NativeMessage('Unknown')],mode=mode)
                self.assertIsNone(result['error'])
                self.assertEqual(result['answer'],case.expected_wrong)
                self.assertEqual(fetches,[])
                self.assertEqual(result['revision']['observations'],[])
                self.assertLessEqual(len(requests),4)

    def test_real_tokenizer_has_a_feasible_native_path_with_intact_packets(self):
        case=list(semantic_cases())[1]
        result,requests,_,_,_,_=self.exercise(case,self.successful_script(case),real_tokenizer=True)
        self.assertIsNone(result['error'])
        self.assertEqual(result['answer'],case.expected_correct)
        self.assertEqual(len(requests),5)
        admission=result['revision']['admission']
        self.assertGreater(admission['judge_prompt_tokens'],100)
        self.assertLessEqual(admission['judge_prompt_tokens'],admission['judge_prompt_bound'])
        self.assertEqual(admission['aux_token_cap'],60000)
        self.assertLessEqual(result['tokens']['total'],200000)

    def test_unknown_judge_after_actually_correct_A_is_an_auxiliary_fatal_loss(self):
        case=next(semantic_cases())
        script=self.successful_script(case,candidate=case.wrong_code)
        bills=[known_usage() for _ in range(5)];bills[-1]=known_usage(known=False)
        result,requests,envelope,_,_,_=self.exercise(case,script,primary_code=case.correct_code,bills=bills)
        self.assertEqual(envelope['payload']['candidate']['released']['answer'],case.expected_correct)
        self.assertIsNone(result['answer'])
        self.assertIsNotNone(result['error'])
        self.assertEqual(len(requests),5)
        self.assertTrue(result['revision']['mechanism']['auxiliary_fatal'])
        self.assertFalse(result['revision']['mechanism']['selected_B'])

    def test_fetch_after_final_in_same_native_batch_invalidates_the_old_B_computation(self):
        case=next(semantic_cases())
        final=native_tool(case.correct_code,identifier='B-before-fetch')
        fetch=native_tool('',name='get_resources_by_patient_fhir_id',identifier='B-later-fetch',
                          resource_type='Observation',patient_fhir_id=PATIENT)
        batch=NativeMessage(json.dumps(counterexample_claim(case)),calls=[
            call.model_dump() for call in [*final.tool_calls,*fetch.tool_calls]])
        script=[native_tool(case.probe_code,mode='inspect',identifier='first-probe'),batch,NativeMessage('Unknown')]
        result,_,envelope,_,_,fetches=self.exercise(case,script)
        self.assertIsNone(result['error'])
        self.assertEqual(result['answer'],case.expected_wrong)
        self.assertEqual(len(fetches),1)
        self.assertIsNone(result['revision']['mechanism']['released_source_bound_B'])
        self.assertEqual(envelope['payload']['candidate']['released']['answer'],case.expected_wrong)


    def test_JSON_only_witness_can_precede_the_complete_alternative_program(self):
        case=next(semantic_cases())
        script=[native_tool(case.probe_code,mode='inspect',identifier='probe'),
                NativeMessage(json.dumps(counterexample_claim(case))),
                native_tool(case.correct_code,identifier='B-after-witness'),adjudication(case)]
        result,requests,_,executions,source_reads,_=self.exercise(case,script)
        self.assertIsNone(result['error'])
        self.assertEqual(result['answer'],case.expected_correct)
        self.assertEqual(len(requests),6)
        self.assertEqual(len(executions),3)  # A final, probe, B final
        self.assertEqual(len(source_reads),2)
        self.assertEqual(result['budget']['tool_calls_run'],5)
        messages=[m if isinstance(m,dict) else m.model_dump() for m in requests[4]['messages']]
        witness_index=next(i for i,m in enumerate(messages)
            if m.get('role')=='assistant'
            and m.get('content')==json.dumps(counterexample_claim(case)))
        self.assertEqual(messages[witness_index+1]['content'],
                         'Witness lineage recorded; this does not prove its semantics.')
        self.assertIn('Now execute the COMPLETE alternative program',messages[witness_index+2]['content'])

    def test_JSON_only_witness_after_source_clean_B_requires_no_reexecution(self):
        case=next(semantic_cases())
        script=[native_tool(case.probe_code,mode='inspect',identifier='probe'),
                native_tool(case.correct_code,identifier='B-before-witness'),
                NativeMessage(json.dumps(counterexample_claim(case))),adjudication(case)]
        result,requests,_,executions,source_reads,_=self.exercise(case,script)
        self.assertIsNone(result['error'])
        self.assertEqual(result['answer'],case.expected_correct)
        self.assertEqual(len(requests),6)
        self.assertEqual(len(executions),3)
        self.assertEqual(len(source_reads),2)
        self.assertEqual(result['budget']['tool_calls_run'],5)

    def test_JSON_only_witness_near_cap_cannot_admit_a_seventh_auxiliary_send(self):
        case=next(semantic_cases())
        witness=lambda:NativeMessage(json.dumps(counterexample_claim(case)))
        script=[native_tool(case.probe_code,mode='inspect',identifier='probe'),witness(),
                native_tool("raise ValueError('recoverable')",identifier='bad-one'),witness(),
                native_tool("raise ValueError('recoverable')",identifier='bad-two'),witness(),
                native_tool(case.correct_code,identifier='unaffordable-finish'),adjudication(case)]
        result,requests,_,_,_,_=self.exercise(case,script)
        self.assertIsNone(result['error'])
        self.assertEqual(result['answer'],case.expected_wrong)
        self.assertLessEqual(len(requests)-2,6)
        self.assertLessEqual(result['budget']['tool_calls_run']-2,6)
        self.assertEqual(result['tokens']['llm_calls'],len(requests))

    def test_established_empty_count_is_distinct_from_a_failed_probe(self):
        resources={'Observation':[observation('other-a',label='other'),observation('other-b',label='other')]}
        count=final_program("sum(r['code']['text']=='target' for r in rows)",fields=('code.text',))
        unfiltered=final_program('len(rows)',fields=('code.text',))
        case=SemanticCase('empty_count','How many of the supplied records are labeled target?',resources,
            unfiltered,count,2,0,'labeled target',
            "answer=[r['code']['text'] for r in retrieved_resources['Observation']]")
        result,requests,_,_,_,_=self.exercise(case,self.successful_script(case)[:2],primary_code=count)
        self.assertIsNone(result['error'])
        self.assertEqual(result['answer'],0)
        self.assertEqual(len(requests),4)
        failed=native_tool("answer=retrieved_resources['UnavailableType']",mode='inspect',identifier='failed-query')
        script=[failed,with_claim(native_tool(unfiltered,identifier='unsupported-B'),counterexample_claim(case)),NativeMessage('Unknown')]
        result,_,_,_,_,_=self.exercise(case,script,primary_code=count)
        self.assertIsNone(result['error'])
        self.assertEqual(result['answer'],0)
        self.assertFalse(result['revision']['observations'][0]['witness_eligible'])
        self.assertIsNone(result['revision']['witness'])

    def test_oversized_observation_is_retained_but_cannot_support_a_witness(self):
        case=next(semantic_cases())
        script=[native_tool("answer='x'*12000",mode='inspect',identifier='large-observation'),
                with_claim(native_tool(case.correct_code,identifier='candidate-B'),counterexample_claim(case)),NativeMessage('Unknown')]
        result,_,_,_,_,_=self.exercise(case,script)
        self.assertIsNone(result['error'])
        self.assertEqual(result['answer'],case.expected_wrong)
        self.assertFalse(result['revision']['observations'][0]['witness_eligible'])
        self.assertIsNone(result['revision']['witness'])


def synthetic_study_records():
    """Hand-built120 metadata cases and480 outcomes, no released QA input."""
    import itertools
    from scripts.v8_mechanism_study import POLICY,QUOTAS
    orders=list(itertools.permutations(('W','R','C')))
    groups=[group for group,quota in QUOTAS.items() for _ in range(quota)]
    cases=[];records=[]
    for index,group in enumerate(groups):
        case={'question_id':f'synthetic-{index:03d}','patient_fhir_id':f'fixture-patient-{index%24:02d}',
              'operation_group':group,'case_index':index+1,'time_block':index//6+1,
              'followup_order':list(orders[index%6]),'template_id':'synthetic-structure-'+group,
              'study_family':group if group in ('lookup_counting','temporal_selection','numerical_aggregation') else 'other'}
        cases.append(case)
        for mode in ('A','W','R','C'):
            records.append({**case,'study_arm':mode,'study_terminal':True,
                'primary_envelope_sha256':'same-A-for-'+case['question_id'],
                'question':'Purely synthetic question','template':'','true_answer':0,'agent_answer':0,
                'evidence':[],'limitation':None,'error':None,
                'mechanism':{'released_source_bound_B':None,'selected_B':False,'auxiliary_fatal':False}})
    return {'policy':deepcopy(POLICY),'questions':120,'cases':cases},records


class MechanismStudyIndependentTests(unittest.TestCase):
    @staticmethod
    def score(answer,reference,*_):
        return {'typed_correct':type(answer) is type(reference) and answer == reference}

    def test_scorer_is_never_called_before_all480_terminal_same_A_outcomes(self):
        from scripts.v8_mechanism_study import analyze_records
        manifest,complete=synthetic_study_records()
        changed_reference=deepcopy(complete);changed_reference[1]['true_answer']=99
        changed_A=deepcopy(complete);changed_A[1]['primary_envelope_sha256']='another-A'
        nonterminal=deepcopy(complete);nonterminal[-1]['study_terminal']=False
        overlapping_effects=deepcopy(complete)
        overlapping_effects[1]['mechanism'].update(selected_B=True,auxiliary_fatal=True)
        cases=[(complete,False),(complete[:-1],True),(complete+[deepcopy(complete[-1])],True),
               (changed_reference,True),(changed_A,True),(nonterminal,True),(overlapping_effects,True)]
        for rows,terminal in cases:
            with self.subTest(rows=len(rows),terminal=terminal):
                scorer=Mock(side_effect=self.score)
                with self.assertRaises(ValueError):
                    analyze_records(rows,manifest,scorer,terminal_confirmed=terminal)
                scorer.assert_not_called()

    def test_help_harm_fatal_gate_algebra_matches_actual_R_minus_A(self):
        from scripts.v8_mechanism_study import analyze_records
        manifest,records=synthetic_study_records()
        indexed={(row['case_index']-1,row['study_arm']):row for row in records}
        # Six corrective opportunities across six patients and two operation groups.
        for index in (0,1,2,30,31,32):
            for mode in ('A','W','R','C'):indexed[index,mode]['agent_answer']=1
            r=indexed[index,'R'];r['agent_answer']=0
            r['mechanism'].update(selected_B=True,released_source_bound_B={
                'answer':0,'evidence':[],'limitation':None,'operation':'Synthetic completed empty count',
                'contributing_ids':[]})
        harmful=indexed[10,'R'];harmful['agent_answer']=1
        harmful['mechanism'].update(selected_B=True,released_source_bound_B={
            'answer':1,'evidence':[],'limitation':None,'operation':'Synthetic scoring-only harmful candidate',
            'contributing_ids':[]})
        fatal=indexed[11,'R'];fatal.update(agent_answer=None,error='Synthetic unknown admitted usage')
        fatal['mechanism']['auxiliary_fatal']=True
        # A computed but unreleased alternative does not create O.
        indexed[3,'R']['mechanism']['computed_withheld_B']={'answer':0}
        result=analyze_records(records,manifest,self.score,terminal_confirmed=True)
        r=result['mechanisms']['R']
        self.assertEqual((r['O'],r['H'],r['M'],r['F'],r['net']),(6,6,1,1,4))
        self.assertEqual(result['R_minus_A_correct'],4)
        self.assertEqual(result['arms']['R']['correct']-result['arms']['A']['correct'],4)
        self.assertTrue(result['advance_gate']['passed'])
        interval=result['paired_question_weighted_patient_bootstrap']['R-A']
        self.assertAlmostEqual(interval['estimate'],4/120)
        self.assertEqual(result['bootstrap']['seed'],5106)

    def test_runner_forks_one_A_without_reference_leakage_or_cache_crosswarming(self):
        from scripts.v8_mechanism_study import run_case
        from types import SimpleNamespace
        cache_module=SimpleNamespace(CACHE_DIR='outside-study',CACHE_ENABLED=False)
        def load_cache(*_a,**_kw):
            path=Path(cache_module.CACHE_DIR)/'A.cache'
            return path.read_bytes() if path.exists() else None
        cache_module.load_from_cache=load_cache
        instances=[];seen=[];primary_runs=[]
        input_data='Question: Synthetic record query without any reference answer.'
        bill={'prompt':100,'completion':20,'cached':0,'usage_available':True}
        class FakeClient:
            def _fetch_resources_with_pagination(self,*_a,**_kw):
                self.last_search_stats={'physical_requests':1,'pages':1,'response_bytes':10,
                                        'returned_records':1,'complete':True}
                return [{}]
        class FakeAgent:
            def __init__(self,mode):self.mode=mode;instances.append(self)
            def run_primary(self,question):
                self_outer.assertEqual(question,input_data)
                self_outer.assertEqual(list(Path(cache_module.CACHE_DIR).iterdir()),[])
                self_outer.assertIsNone(cache_module.load_from_cache('synthetic'))
                (Path(cache_module.CACHE_DIR)/'A.cache').write_bytes(b'post-A-bytes')
                FakeClient()._fetch_resources_with_pagination()
                primary_runs.append(1)
                return {'immutable_A':'synthetic','private_mutations':[]}
            def run_from_primary(self,question,envelope,mode):
                self_outer.assertEqual(question,input_data)
                self_outer.assertEqual(envelope,{'immutable_A':'synthetic','private_mutations':[]})
                seen.append((mode,deepcopy(envelope)))
                envelope['private_mutations'].append(mode)
                if mode!='A':
                    self_outer.assertEqual(cache_module.load_from_cache('synthetic'),b'post-A-bytes')
                    self_outer.assertEqual(sorted(path.name for path in Path(cache_module.CACHE_DIR).iterdir()),['A.cache'])
                    (Path(cache_module.CACHE_DIR)/(mode+'.cache')).write_bytes(mode.encode())
                    if mode!='W':FakeClient()._fetch_resources_with_pagination()
                calls=[deepcopy(bill)]+([] if mode=='A' else [deepcopy(bill)])
                return {'answer':7,'evidence':[],'limitation':None,'error':None,
                        'tokens':{'per_call':calls},'revision':{'mechanism':{
                            'released_source_bound_B':None,'selected_B':False,'auxiliary_fatal':False}}}
        self_outer=self
        def parse(raw,**_):
            value=deepcopy(raw);value['agent_answer']=value.pop('answer');return value
        def externalize(row,*_a,**_kw):return row
        case={'question_id':'synthetic-runner-case','patient_fhir_id':'synthetic-patient',
              'followup_order':['C','R','W']}
        delivered=[]
        with tempfile.TemporaryDirectory() as tmp:
            rows=run_case(input_data,case,tmp,agent_factory=FakeAgent,parse_output=parse,
                externalize=externalize,reference_metadata={'question':input_data,
                    'true_answer':'RECORDER_ONLY_REFERENCE_SENTINEL'},cache_module=cache_module,
                client_class=FakeClient,on_record=delivered.append)
        self.assertEqual(primary_runs,[1])
        self.assertEqual([mode for mode,_ in seen],['A','C','R','W'])
        self.assertEqual(len(instances),5)
        self.assertEqual(len(delivered),4)
        self.assertEqual(len({row['primary_envelope_sha256'] for row in rows}),1)
        self.assertEqual([row['physical_usage']['observed_invocations'] for row in rows],[1,1,1,1])
        self.assertEqual([row['physical_fhir']['observed_physical_queries'] for row in rows],[1,1,1,0])
        self.assertTrue(all(row['cache_activity']['hits']==1 for row in rows[1:]))
        self.assertEqual((cache_module.CACHE_DIR,cache_module.CACHE_ENABLED),('outside-study',False))

    def test_real_isolated_CLI_analyzes_only_saved_synthetic480(self):
        import subprocess
        import sys
        manifest,records=synthetic_study_records()
        # The released typed scorer consumes serialized SQL-row references.
        for row in records:row['true_answer']='[[0]]'
        script=Path(__file__).resolve().parents[1]/'scripts/v8_mechanism_study.py'
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            manifest_path=directory/'manifest.json';records_path=directory/'records.json'
            result_path=directory/'analysis.json'
            manifest_path.write_text(json.dumps(manifest))
            records_path.write_text(json.dumps(records))
            command=[sys.executable,'-IB',str(script),'analyze','--manifest',str(manifest_path),
                     '--input',str(records_path),'--output',str(result_path),'--terminal-confirmed']
            environment={**os.environ,'FHIR_LOCAL_AZURE_CONFIG':'','FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR':''}
            completed=subprocess.run(command,capture_output=True,text=True,env=environment,timeout=30)
            self.assertEqual(completed.returncode,0,completed.stderr)
            result=json.loads(result_path.read_text())
            self.assertEqual(result['arms']['A']['correct'],120)
            self.assertEqual(result['arms']['R']['correct'],120)
            self.assertFalse(result['advance_gate']['passed'])
            result_path.unlink();records_path.write_text(json.dumps(records[:-1]))
            blocked=subprocess.run(command,capture_output=True,text=True,env=environment,timeout=30)
            self.assertNotEqual(blocked.returncode,0)
            self.assertFalse(result_path.exists())
            self.assertIn('480',blocked.stderr)

    def test_runtime_input_keeps_scoring_template_recorder_side_and_excludes_SQL(self):
        import csv
        from scripts.v8_mechanism_study import _runtime_inputs
        with tempfile.TemporaryDirectory() as tmp:
            dataset=Path(tmp)/'synthetic_inputs.csv';families=Path(tmp)/'synthetic_families.csv'
            with dataset.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['question_id','patient_fhir_id','question','true_answer','assumption','sql'])
                writer.writeheader()
                writer.writerow({'question_id':'chosen','patient_fhir_id':'p-fixture','question':'Synthetic question',
                    'true_answer':'0','assumption':'Synthetic context','sql':'SQL_SENTINEL_NEVER_FORWARD'})
                writer.writerow({'question_id':'not-chosen','patient_fhir_id':'p-other','question':'Unselected synthetic text',
                    'true_answer':'999','assumption':'','sql':'UNSELECTED_SQL_SENTINEL'})
            with families.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['question_id','template']);writer.writeheader()
                writer.writerow({'question_id':'chosen','template':'Generic selected fixture template'})
                writer.writerow({'question_id':'not-chosen','template':'Other synthetic template'})
            selected=_runtime_inputs(dataset,[{'question_id':'chosen','patient_fhir_id':'p-fixture'}],families=families)
        self.assertEqual(set(selected),{'chosen'})
        self.assertEqual(selected['chosen']['template'],'Generic selected fixture template')
        self.assertNotIn('sql',selected['chosen'])
        self.assertNotIn('SQL_SENTINEL',json.dumps(selected))
        self.assertNotIn('Unselected synthetic text',json.dumps(selected))
