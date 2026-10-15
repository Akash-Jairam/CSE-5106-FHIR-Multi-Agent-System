"""V8: full natural Single, then a bounded, draft-conditioned investigator.

W/R/C differ only in retrieval permission or retained conversation context. A is
immutable and every policy is conceptually charged its full primary. Executed
witnesses and source checks bind provenance; neither is a semantic oracle.
"""
from copy import deepcopy
import hashlib
import json

from .base_agent import BaseAgent
from .controller import FixedController
from .framework import RETRIEVAL_TOOLS
from .single_agent import SingleAgent
from .residual_trajectory_agent import candidate_entry, selection_object, rendered
from .residual_trajectory_phase import (fingerprint, json_text, message_dict,
    merge_ledger, reserve, same, strict_object)
from .counterexample_investigator_phase import (AUX_SDK, AUX_TOKENS, AUX_TOOLS,
    GLOBAL_SDK, JUDGE_COMPLETION, base_module, bounded_call, caps,
    capture_primary, restore_primary, snapshot)
from utils import parse_question_context

PROTOCOL = 'full-single-counterexample-investigator-v8'
ENTRY_BYTES = 16384
OBSERVATION_BYTES = 8192
WITNESS_BYTES = 4096
FRAMING_BOUND = 512
MAX_PROBES = 2
PROBE_COMPLETION = 2048
FINAL_COMPLETION = 4096
RENDERED = ('answer', 'evidence', 'limitation')

INVESTIGATION_TASK = """Investigate the EXISTING A program against the literal original question/context.
First identify a concrete question obligation, then seek an executed counterexample
to what A actually does. You are conditioned on A; this is not independent proof.
Use at most TWO combined inspect/retrieval interactions. A small new derived
observation of already held records can be useful; new retrieval is optional when
permitted. Do not blindly rerun A or merely assert a better answer. Check temporal
ties, scope, concepts, requested fields and complete operands when relevant.
All held data is in retrieved_resources, initially a private copy of A's workspace.
Do not mutate the workspace in inspection code; retrieval alone may replace entries.
Retrieval invalidates YOUR prior final computation, never A. Every Python invocation
is a fresh worker: define every needed variable again. Full audit replies may exceed
model previews; an omitted/truncated observation cannot support a witness.

After a successful inspect/retrieval, the host reports observation_1 or observation_2
bound to its actual arguments/reply, workspace and A-program SHA. To propose B put
one finite JSON object in assistant content (native tool calls can accompany it):
{"obligation_span":"exact literal quote from original question/context",
 "observation_id":"observation_1", "A_behavior":"concrete behavior of A's actual program",
 "contradiction":"why that ACTUAL observation conflicts with that obligation"}.
Reference only an observation you have already received. Never invent a field value,
observation, hidden expected answer, or critique without executing a real probe.
Quotation/provenance binding does not establish that your interpretation is correct.

If useful counterevidence exists, execute a COMPLETE alternative program with
execute_python_code(mode="final") setting answer, evidence, contributing_ids,
operation, limitation in that SAME execution. It must recompute the whole answer,
not patch A or merge unsupported evidence. Use the unchanged native tool contract.
The host immediately runs the ORIGINAL source-field check after a successful full
program, WITHOUT an extra prose-finalization request. Source/program errors can be
repaired with another complete program only while the reserved path fits. At most
two automatic checks are available. At most five investigation sends plus one
fresh tool-free judge, six tools, and 60000 auxiliary tokens are available, subject
to the actual residual global budget. The host reserves the finishing path first.
Do not spend remaining budget to manufacture disagreement. When evidence or budget
is inconclusive stop; the whole actual A will be preserved. No new answer is needed
in prose. A valid, source-bound B and an actual witness still require a fresh judge.
"""

JUDGE_TASK = """Judge EXISTING A and B for the literal original question/context.
Return only an existing choice; no new code, answer, retrieval, tools or evidence
merge. A is a full natural Single result. B is a draft-conditioned investigation,
not an independent sample. Read their complete executed programs, released values,
source checks, limitations and the actual recorded counterexample observation.
The witness quotes a literal obligation and binds an actual observation to A's
program/workspace, but its claimed contradiction is fallible model prose. Source
checks validate source lineage, not semantics or completed search. Successful
retrieval/count metadata is not completeness proof. Do not infer omitted facts.
For each material question requirement compare concrete program/field facts using
supported, not_found or contradicted. Absence alone is not contradiction. B may be
selected only if all its requirements are supported, at least one A requirement is
concretely contradicted by the recorded observation, and the quoted obligation is
material to the actual question. Otherwise choose A or UNKNOWN. Never choose B
merely for being different, longer, nonempty or source-checked. Return finite JSON:
{"requirements":[{"requirement":"question obligation","A":{"status":"supported|not_found|contradicted","basis":"concrete facts"},"B":{"status":"supported|not_found|contradicted","basis":"concrete facts"}}],"choice":"A|B|UNKNOWN","reason":"concrete comparison","obligation_span":"exact supplied witness quote","observation_id":"exact supplied observation ID"}.
"""


def workspace_inventory(workspace):
    """Counts only; all held records remain accessible to native Python tools."""
    return [{'resource_type': name, 'held_count': len(records)}
        for name, records in sorted(workspace.items()) if isinstance(records, list)]


def witness_object(content, observations, question, context, a):
    value = strict_object(content, WITNESS_BYTES)
    keys = {'obligation_span', 'observation_id', 'A_behavior', 'contradiction'}
    if set(value) != keys or any(not isinstance(value[k], str) or not value[k].strip() for k in keys):
        raise ValueError('witness schema differs')
    if any(len(value[k]) > limit for k, limit in
            [('obligation_span', 1000), ('A_behavior', 1500), ('contradiction', 1500)]):
        raise ValueError('witness prose exceeds coverage bound')
    # Literal original text only; model-generated summaries cannot supply a span.
    if value['obligation_span'] not in question:
        raise ValueError('obligation is not a literal original question/context span')
    observation = next((item for item in observations
        if item['observation_id'] == value['observation_id']), None)
    if observation is None or not observation['witness_eligible']:
        raise ValueError('witness references no prior complete successful observation')
    bound = {**value, 'observation_sha256': fingerprint(observation),
        'A_program_sha256': a['binding']['code_sha256'],
        'A_workspace_sha256': a['workspace_sha256'],
        'A_snapshot_sha256': fingerprint(a)}
    if len(json_text(bound).encode()) > WITNESS_BYTES:
        raise ValueError('witness entry exceeds intact coverage bound')
    return bound


def judge_object(response, witness):
    message = message_dict(response)
    if message.get('tool_calls'):
        raise ValueError('judge returned native tools')
    value = strict_object(message.get('content'), 20000)
    if set(value) != {'requirements', 'choice', 'reason', 'obligation_span', 'observation_id'}:
        raise ValueError('judge schema differs')
    if (value['obligation_span'] != witness['obligation_span']
            or value['observation_id'] != witness['observation_id']):
        raise ValueError('judge witness linkage differs')
    basic = {key: value[key] for key in ('requirements', 'choice', 'reason')}
    selection_object({'content': json_text(basic)})
    return value


def _parse_reply(reply):
    content = reply.get('content') if isinstance(reply, dict) else None
    try:
        return json.loads(content)
    except (ValueError, TypeError):
        return None


def _observation(call, arguments, reply, before, after, a, ordinal):
    output = _parse_reply(reply)
    content = reply.get('content', '')
    is_retrieval = call.function.name in RETRIEVAL_TOOLS
    success = (isinstance(output, dict) and not output.get('error'))
    if is_retrieval and output is None:
        # The unchanged native retrieval tools return an explicit count receipt.
        import re
        success = bool(re.fullmatch(r"retrieved_resources\['[^']+'\] \(\d+ items; \d+ fetched\)", content))
    truncated = isinstance(output, dict) and any(output.get(key) for key in
        ('truncated', 'stdout_truncated', 'answer_truncated', 'output_truncated'))
    value = {'observation_id': 'observation_' + str(ordinal),
        'tool_call_id': call.id, 'tool': call.function.name,
        'arguments': deepcopy(arguments), 'actual_reply': deepcopy(reply),
        'workspace_before_sha256': before, 'workspace_after_sha256': after,
        'A_program_sha256': a['binding']['code_sha256'],
        'A_workspace_sha256': a['workspace_sha256'],
        'assurance': 'actual_call_and_observation_lineage_not_semantic_proof',
        'witness_eligible': bool(success and not truncated)}
    if len(json_text(value).encode()) > OBSERVATION_BYTES:
        value['witness_eligible'] = False
        value['coverage_reason'] = 'complete observation exceeds intact judge bound'
    return value


class CounterexampleInvestigatorAgent(BaseAgent):
    def __init__(self, model, verbose=False, base_url=None, mode='R'):
        super().__init__(model, verbose=verbose, base_url=base_url)
        if mode not in {'A', 'W', 'R', 'C'}:
            raise ValueError('V8 mode must be A, W, R or C')
        self.mode = mode
        self._prepared_primary = None

    def run_primary(self, question):
        """One actual natural Single; caller owns per-case cache isolation."""
        return capture_primary(self, question)

    def run_from_primary(self, question, envelope, mode=None):
        if mode is not None and mode not in {'A', 'W', 'R', 'C'}:
            raise ValueError('V8 mode must be A, W, R or C')
        self._prepared_primary = deepcopy(envelope)
        previous = self.mode
        self.mode = mode or self.mode
        try:
            return self.run(question)
        finally:
            self._prepared_primary = None
            self.mode = previous

    def _aggregate_error(self):
        if (self.total_usage['total_tokens'] > self.max_total_tokens
                or self.tool_calls_made > self.max_tool_calls
                or self.total_usage['llm_calls'] > GLOBAL_SDK):
            return 'actual global ceiling exceeded'
        return None

    def _release(self, snapshots, phases, status, *, chosen='A', admission=None,
                 observations=None, witness=None, selection=None, fatal=None):
        selected = snapshots[chosen]
        released = selected['released']
        self.budget_stop, self.last_llm_error = None, None
        index = {key: fingerprint(value) for key, value in snapshots.items()}
        revision = {'protocol': PROTOCOL, 'mode': self.mode, 'status': status,
            'selected': chosen, 'fatal': bool(fatal),
            'A_only': {'available': True, 'snapshot_sha256': index['A'],
                'error': snapshots['A']['released'].get('error'),
                'source_bound': snapshots['A']['artifact'] is not None},
            'candidate_snapshot_sha256': index, 'phases': deepcopy(phases),
            'admission': deepcopy(admission), 'observations': deepcopy(observations or []),
            'witness': deepcopy(witness), 'selection': deepcopy(selection),
            'messages': [deepcopy(value) for value in snapshots.values()],
            'final_check_issues': [] if selected['artifact'] is not None and not fatal else None,
            'alternative_selected': chosen == 'B',
            'auxiliary_policy': 'immediate_original_source_check_after_full_program_no_prose_finalization',
            'assurance': 'execution_source_and_literal_binding_not_semantic_proof'}
        b = snapshots.get('B')
        revision['mechanism'] = {
            'released_source_bound_B': ({key: deepcopy(b['artifact'][key]) for key in
                ('answer', 'evidence', 'operation', 'contributing_ids', 'limitation')}
                if b is not None and b['artifact'] is not None else None),
            'selected_B': chosen == 'B' and not bool(fatal),
            'auxiliary_fatal': bool(fatal) and status != 'fatal_primary',
            'status': status, 'witness_bound': witness is not None,
            'probe_interactions': len(observations or [])}
        if fatal:
            return {'answer': None, 'evidence': [], 'limitation': fatal,
                'error': 'V8 execution/accounting failure: ' + fatal, 'trace': self.messages,
                'retrieved_fhir_resources': deepcopy(selected['workspace']), 'revision': revision}
        return {**{key: deepcopy(released.get(key)) for key in RENDERED},
            'error': released.get('error'), 'trace': self.messages,
            'retrieved_fhir_resources': deepcopy(selected['workspace']), 'revision': revision}

    def _run(self, question):
        self.set_budget(**caps(self))
        self._v7_merged_phases = set()
        envelope = self._prepared_primary if self._prepared_primary is not None else self.run_primary(question)
        a, payload = restore_primary(self, question, envelope)
        frozen_a = fingerprint(a)
        snapshots, phases = {'A': a}, [deepcopy(a['diagnostics'])]
        if a['diagnostics']['fatal'] or self._aggregate_error():
            return self._release(snapshots, phases, 'fatal_primary',
                fatal=self._aggregate_error() or 'primary accounting unverifiable')
        if self.mode == 'A':
            return self._release(snapshots, phases, 'primary_only')
        if a['artifact'] is None:
            return self._release(snapshots, phases, 'primary_ineligible')
        try:
            entry_a = candidate_entry(a, payload['trace'])
            if len(json_text(entry_a).encode()) > ENTRY_BYTES:
                return self._release(snapshots, phases, 'A_entry_coverage_skip')
            context = FixedController.question_context(question)
            packet = {'context': context, 'A': entry_a,
                'workspace_inventory': workspace_inventory(a['workspace']),
                'retrieval_permitted': self.mode != 'W'}
            judge_base = {'context': context, 'A': entry_a, 'B': None,
                          'witness': None, 'observation': None}
            judge_messages = [{'role': 'system', 'content': JUDGE_TASK},
                {'role': 'user', 'content': json_text(judge_base)}]
            # ASCII serialized bytes upper-bound future tokenized entries. No
            # candidate, witness or observation is shortened to fit this bound.
            b_entry_bound = min(ENTRY_BYTES, max(4096, len(json_text(entry_a).encode()) + 4096))
            judge_bound = (base_module.count_tokens_in_messages(judge_messages)
                + b_entry_bound + OBSERVATION_BYTES + WITNESS_BYTES + FRAMING_BOUND)
            judge_reservation = reserve(judge_bound) + JUDGE_COMPLETION
            aux_tokens = min(AUX_TOKENS, self.max_total_tokens - self.total_usage['total_tokens'])
            aux_tools = min(AUX_TOOLS, self.max_tool_calls - self.tool_calls_made)
            aux_sdk = min(AUX_SDK, GLOBAL_SDK - self.total_usage['llm_calls'])
            admission = {'aux_token_cap': aux_tokens, 'aux_tool_cap': aux_tools,
                'aux_sdk_cap': aux_sdk, 'judge_prompt_bound': judge_bound,
                'judge_token_reservation': judge_reservation,
                'A_entry_bytes': len(json_text(entry_a).encode()),
                'future_B_entry_byte_bound': b_entry_bound,
                'observation_byte_bound': OBSERVATION_BYTES,
                'witness_byte_bound': WITNESS_BYTES, 'max_probe_interactions': MAX_PROBES}
            if aux_tokens <= judge_reservation or aux_tools < 3 or aux_sdk < 3:
                return self._release(snapshots, phases, 'residual_coverage_skip', admission=admission)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            return self._release(snapshots, phases, 'A_packet_unavailable',
                admission={'exception_class': type(exc).__name__})

        local = SingleAgent(self.model, max_iterations=aux_sdk, verbose=self.verbose, base_url=self.base_url)
        local.set_budget(max_total_tokens=aux_tokens, max_tool_calls=aux_tools)
        local.set_perturbation(self.perturbation)
        local._question, local._patient_fhir_id, local._current_time = parse_question_context(question)
        local.final_computation, local.last_execution = None, None
        local._requires_final_computation, local._final_computation_error = False, None
        workspace = deepcopy(a['workspace'])
        investigation_message = {'role': 'user', 'content': INVESTIGATION_TASK + '\n' + json_text(packet)}
        initial = deepcopy(payload['model_messages'] if self.mode == 'C' else local.system_msg)
        # Same user instruction and priority in R/C; C additionally retains A's
        # actual model context. No forced compaction, summary or dropped history.
        # The B execution namespace excludes A's old automatic-check IDs; C's
        # MODEL context still retains every original A message intact.
        local.messages = deepcopy(local.system_msg) + [deepcopy(investigation_message)]
        local.model_messages = deepcopy(initial) + [deepcopy(investigation_message)]
        state = {'fatal': False, 'errors': [], 'calls': [], 'sdk_cap': aux_sdk,
            'probe_interactions': 0, 'automatic_checks': 0, 'stop': None}
        observations, witness, selection = [], None, None
        released = {'answer': None, 'evidence': [], 'limitation': 'No eligible complete alternative',
            'error': None, 'retrieved_fhir_resources': workspace}
        tools = [tool for tool in local.tools if self.mode != 'W'
            or tool['function']['name'] not in RETRIEVAL_TOOLS]
        allowed_names = {tool['function']['name'] for tool in tools}
        try:
            while local.total_usage['llm_calls'] < aux_sdk - 1:
                # At least a whole final execution and its source check must
                # remain before spending another investigation request.
                if local.max_tool_calls - local.tool_calls_made < 2:
                    state['stop'] = 'whole-program/source-check path unavailable'
                    break
                if state['automatic_checks'] >= 2:
                    state['stop'] = 'no remaining automatic source-check path'
                    break
                prompt = base_module.count_tokens_in_messages(local.model_messages, tools)
                final_request_reserve = reserve(prompt + PROBE_COMPLETION + OBSERVATION_BYTES
                    + FRAMING_BOUND) + FINAL_COMPLETION
                remaining_tokens = aux_tokens - local.total_usage['total_tokens']
                can_probe = (state['probe_interactions'] < MAX_PROBES
                    and aux_sdk - local.total_usage['llm_calls'] >= 3
                    and aux_tools - local.tool_calls_made >= 3
                    and remaining_tokens >= judge_reservation + final_request_reserve
                        + reserve(prompt) + PROBE_COMPLETION)
                has_observation = any(item['witness_eligible'] for item in observations)
                if not has_observation and not can_probe:
                    state['stop'] = 'initial probe plus intact final request/judge path unavailable'
                    break
                response, error = bounded_call(local, local.model_messages, tools, state,
                    reserve_tokens=judge_reservation + (final_request_reserve if can_probe else 0),
                    reserve_sdk=2 if can_probe else 1,
                    completion=PROBE_COMPLETION if can_probe else FINAL_COMPLETION)
                if response is None:
                    break
                local._append_message(response)
                content = message_dict(response).get('content')
                witness_received = False
                if isinstance(content, str) and content.strip():
                    try:
                        witness = witness_object(content, observations, question, context, a)
                        witness_received = True
                        local._append_message({'role': 'user', 'content': 'Witness lineage recorded; this does not prove its semantics.'})
                    except (ValueError, TypeError, KeyError) as exc:
                        local._append_message({'role': 'user', 'content':
                            'No witness admitted (' + type(exc).__name__ + '). Use the exact metadata schema and a prior real observation.'})
                calls = getattr(response, 'tool_calls', None) or []
                if not calls:
                    computed = local._computed_candidate(workspace)
                    clean_released_b = (isinstance(computed, dict) and not computed.get('error')
                        and (released.get('revision') or {}).get('final_check_issues') == []
                        and same({key: computed[key] for key in RENDERED},
                                 {key: released.get(key) for key in RENDERED}))
                    if witness_received and not clean_released_b:
                        if local.total_usage['llm_calls'] >= aux_sdk - 1:
                            state['stop'] = 'witness received but complete-program/judge SDK path exhausted'
                            break
                        local._append_message({'role': 'user', 'content':
                            'Witness lineage is recorded. Now execute the COMPLETE alternative program with '
                            'mode=final and all five result variables. The original source check and fresh '
                            'judge must still fit the unchanged remaining budget.'})
                        continue
                    state['stop'] = 'investigator stopped without a new complete program'
                    break
                new_observations = []
                for call in calls:
                    try:
                        arguments = json.loads(call.function.arguments)
                        if not isinstance(arguments, dict):
                            raise ValueError('tool arguments must be an object')
                    except (ValueError, TypeError):
                        arguments = {}
                    name = call.function.name
                    is_probe = name in RETRIEVAL_TOOLS or (name == 'execute_python_code'
                        and arguments.get('mode', 'auto') != 'final')
                    refusal = None
                    if name not in allowed_names:
                        refusal = 'tool is not in the exact advertised native tool catalogue'
                    elif name in RETRIEVAL_TOOLS and self.mode == 'W':
                        refusal = 'retrieval is not permitted in W'
                    elif is_probe and not can_probe:
                        refusal = 'probe plus intact final request/judge path is not reserved'
                    elif is_probe and state['probe_interactions'] >= MAX_PROBES:
                        refusal = 'two probe/acquisition interactions already attempted'
                    elif is_probe and local.max_tool_calls - local.tool_calls_made < 3:
                        refusal = 'reserved final execution/source-check path would be consumed'
                    elif not is_probe and local.max_tool_calls - local.tool_calls_made < 2:
                        refusal = 'reserved source-check path would be consumed'
                    if refusal:
                        local._append_message({'role': 'tool', 'tool_call_id': call.id,
                            'name': name, 'content': 'Not run: V8 admission: ' + refusal})
                        if (name not in allowed_names or name in RETRIEVAL_TOOLS
                                or (name == 'execute_python_code'
                                    and arguments.get('mode', 'auto') != 'inspect')):
                            local._invalidate_computation('A requested mutation was refused before dispatch')
                        state['stop'] = refusal
                        continue
                    if not local._take_tool_call():
                        local._append_message(local._tool_budget_message(call))
                        state['stop'] = 'auxiliary tool allowance exhausted'
                        continue
                    if is_probe:
                        state['probe_interactions'] += 1
                    before = SingleAgent._workspace_sha256(workspace)
                    try:
                        reply = local._handle_tool_call(call, workspace)
                        if reply is None:
                            reply = {'role': 'tool', 'tool_call_id': call.id, 'name': name,
                                     'content': json_text({'error': 'Unknown native tool'})}
                    except Exception as exc:
                        # A charged handler failure is honest known inability,
                        # not a fabricated observation or another SDK attempt.
                        reply = {'role': 'tool', 'tool_call_id': call.id, 'name': name,
                            'content': json_text({'error': 'Native tool raised ' + type(exc).__name__})}
                        local._invalidate_computation('Native tool execution failed')
                    after = SingleAgent._workspace_sha256(workspace)
                    local._append_message(reply)
                    if is_probe:
                        observation = _observation(call, arguments, reply, before, after, a,
                                                   state['probe_interactions'])
                        # Code-side workspace mutation is not a legitimate
                        # retrieval. Reject its witness and derived computation.
                        if name == 'execute_python_code' and before != after:
                            observation['witness_eligible'] = False
                            observation['coverage_reason'] = 'inspection mutated workspace'
                            local._invalidate_computation('Inspection mutated the copied workspace')
                        observations.append(observation)
                        new_observations.append(observation)
                for observation in new_observations:
                    local._append_message({'role': 'user', 'content': json_text({
                        'observation_id': observation['observation_id'],
                        'observation_sha256': fingerprint(observation),
                        'witness_eligible': observation['witness_eligible'],
                        'assurance': observation['assurance']})})
                computed = local._computed_candidate(workspace)
                if computed and not computed.get('error'):
                    if state['automatic_checks'] >= 2:
                        state['stop'] = 'two automatic source checks already attempted'
                        break
                    state['automatic_checks'] += 1
                    output = tuple(computed[key] for key in RENDERED)
                    issues = local._automatic_check(output, state['automatic_checks'])
                    if not issues:
                        released = {**{key: deepcopy(computed[key]) for key in RENDERED},
                            'error': None, 'retrieved_fhir_resources': workspace,
                            'revision': local._revision(output, [], output, [])}
                        if witness is not None:
                            break
                        local._append_message({'role': 'user', 'content':
                            'The program/source check passed, but a prior real counterexample and exact witness metadata are still required. No candidate may switch without them.'})
                    else:
                        local._invalidate_computation('Original source check requested a complete revised program')
                        local._append_message({'role': 'user', 'content':
                            'Original source check problems: ' + json_text(issues) +
                            '. Correct the evidence/program and execute the complete final program again if the reserved path fits.'})
                elif local._final_computation_error:
                    local._append_message({'role': 'user', 'content':
                        'Complete final program unavailable: ' + local._final_computation_error})
                if state['stop'] or state['fatal']:
                    break
        except Exception as exc:
            state['stop'] = 'auxiliary orchestration failed: ' + type(exc).__name__
            local._invalidate_computation(state['stop'])

        released['trace'] = local.messages
        released['retrieved_fhir_resources'] = workspace
        b, b_trace, _ = snapshot(local, released, 'counterexample_investigation', 'B',
            aux_tokens, aux_tools, aux_sdk, state['errors'], state)
        # A source-bound computed candidate can be diagnostically present, but
        # only a lineage-bound real observation makes it switch-eligible.
        b['diagnostics'].update(probe_interactions=state['probe_interactions'],
            automatic_checks=state['automatic_checks'], witness_bound=witness is not None,
            investigation_stop=state['stop'])
        snapshots['B'] = b
        self.messages.extend(b_trace)
        if state['fatal'] or b['diagnostics']['fatal']:
            self._merge_aux(local, phases, state, aux_tokens, aux_tools, aux_sdk)
            return self._release(snapshots, phases, 'fatal_investigation', admission=admission,
                observations=observations, witness=witness, fatal='auxiliary accounting unverifiable')
        if b['artifact'] is None or witness is None:
            self._merge_aux(local, phases, state, aux_tokens, aux_tools, aux_sdk)
            return self._release(snapshots, phases, 'B_ineligible', admission=admission,
                observations=observations, witness=witness,
                fatal=self._aggregate_error() or ('auxiliary accounting unverifiable' if state['fatal'] else None))
        if same(a['released'].get('answer'), b['released'].get('answer')):
            self._merge_aux(local, phases, state, aux_tokens, aux_tools, aux_sdk)
            return self._release(snapshots, phases, 'agreement_preserved', admission=admission,
                observations=observations, witness=witness,
                fatal=self._aggregate_error() or ('auxiliary accounting unverifiable' if state['fatal'] else None))
        try:
            entry_b = candidate_entry(b, b_trace)
            if len(json_text(entry_b).encode()) > b_entry_bound:
                raise ValueError('B entry exceeds intact coverage bound')
            observation = next(item for item in observations
                               if item['observation_id'] == witness['observation_id'])
            if fingerprint(observation) != witness['observation_sha256']:
                raise ValueError('witness observation changed')
            judge_packet = {**judge_base, 'B': entry_b, 'witness': witness, 'observation': observation}
            actual_messages = [{'role': 'system', 'content': JUDGE_TASK},
                {'role': 'user', 'content': json_text(judge_packet)}]
            actual_prompt = base_module.count_tokens_in_messages(actual_messages)
            admission.update(judge_prompt_tokens=actual_prompt,
                judge_packet_sha256=fingerprint(judge_packet),
                B_entry_bytes=len(json_text(entry_b).encode()))
            if actual_prompt > judge_bound or reserve(actual_prompt) + JUDGE_COMPLETION > judge_reservation:
                raise ValueError('actual judge request exceeds prospective reservation')
            self.messages.extend([{**deepcopy(item), 'phase': 'counterexample_judge'} for item in actual_messages])
            response, error = bounded_call(local, actual_messages, None, state,
                completion=JUDGE_COMPLETION, phase='counterexample_judge')
            status = 'judge_unavailable'
            allow_b = False
            if response is not None:
                self.messages.append({**message_dict(response), 'phase': 'counterexample_judge'})
                selection = judge_object(response, witness)
                allow_b = (selection['choice'] == 'B'
                    and all(item['B']['status'] == 'supported' for item in selection['requirements'])
                    and any(item['A']['status'] == 'contradicted' for item in selection['requirements']))
                status = 'selected_B' if allow_b else 'judge_preserved_A'
        except (ValueError, TypeError, KeyError, AttributeError, StopIteration) as exc:
            status, allow_b = 'judge_malformed_or_coverage_skip', False
            admission['judge_exception_class'] = type(exc).__name__
        self._merge_aux(local, phases, state, aux_tokens, aux_tools, aux_sdk)
        fatal = self._aggregate_error() or ('auxiliary accounting unverifiable' if state['fatal'] else None)
        if fingerprint(a) != frozen_a:
            fatal = 'immutable primary snapshot changed'
        return self._release(snapshots, phases, 'fatal_auxiliary' if fatal else status,
            chosen='B' if allow_b and not fatal else 'A', admission=admission,
            observations=observations, witness=witness, selection=selection, fatal=fatal)

    def _merge_aux(self, local, phases, state, token_cap, tool_cap, sdk_cap):
        from .residual_trajectory_phase import validated_ledger
        scratch, raw, problems, unknown = validated_ledger(local)
        errors = [*state['errors'], *problems]
        if unknown:
            errors.append('auxiliary provider usage unknown')
        if (scratch.total_usage['total_tokens'] > token_cap or scratch.tool_calls_made > tool_cap
                or scratch.total_usage['llm_calls'] > sdk_cap):
            errors.append('auxiliary actual ceiling overshoot')
        errors.extend(merge_ledger(self, scratch, 'counterexample_auxiliary'))
        state['fatal'] = bool(errors)
        state['errors'] = errors
        phases.append({'phase': 'counterexample_auxiliary', 'tokens': scratch.total_usage['total_tokens'],
            'tool_calls': scratch.tool_calls_made, 'model_calls': scratch.total_usage['llm_calls'],
            'token_cap': token_cap, 'tool_cap': tool_cap, 'sdk_cap': sdk_cap,
            'fatal': bool(errors), 'usage_known': not unknown, 'errors': errors,
            'raw_accounting': raw, 'calls': deepcopy(state['calls']),
            'probe_interactions': state['probe_interactions'],
            'automatic_checks': state['automatic_checks'], 'stop': state['stop']})


def run_v8_job(args_tuple):
    from utils import run_agent_safe
    import tools.cache as cache_module
    input_data, strategy, model, verbose, base_url, enable_cache, *rest = args_tuple
    if strategy != 'multi_agent_v8':
        raise ValueError('V8 worker received a different strategy')
    cache_module.CACHE_ENABLED = enable_cache
    agent = CounterexampleInvestigatorAgent(model, verbose=verbose, base_url=base_url)
    agent.set_budget(**((rest[0] if rest else None) or {}))
    agent.set_perturbation(rest[1] if len(rest) > 1 else None)
    include_full_audit = bool(rest[2]) if len(rest) > 2 else False
    audit_context = rest[3] if len(rest) > 3 else None
    if include_full_audit:
        output = run_agent_safe(agent, input_data, include_full_audit=True)
        if audit_context is not None:
            from scripts.audit_store import externalize_result_audit
            output = externalize_result_audit(output, audit_context['output_dir'],
                                             question_id=audit_context['question_id'])
        return output
    return run_agent_safe(agent, input_data)
