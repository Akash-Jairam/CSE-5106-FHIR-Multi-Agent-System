"""V3 production integration: mocked LLM/store, real sandbox and source checks."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent.query_contract_agent import QueryContractAgent, PhaseHost, PHASE_CAPS, LedgerSolver, parse_contract, patched_program
from tests.test_phase2_verification import PATIENT, NOW, call, fetch, reply, observation
from tools import cache

BARE = "How many weight measurements were greater than 70?"
QUESTION = f"Question: {BARE}\nContext:\nPatient FHIR ID is {PATIENT}.\nAssume current time is {NOW}."
STORE = [observation("o1", 70), observation("o2", 71)]
PACKET = {"resources": {}, "text": "Starting packet is empty; fetch required records."}
EXPRESSION = "r['valueQuantity']['value'] >= 70"
CODE = """rows = [r for r in retrieved_resources['Observation'] if r['valueQuantity']['value'] >= 70]
answer = len(rows)
evidence = [{'resource_id': 'Observation/' + r['id'], 'field_path': 'valueQuantity.value', 'value': r['valueQuantity']['value']} for r in rows]
contributing_ids = ['Observation/' + r['id'] for r in rows]
operation = 'count selected weight measurements'
limitation = None
"""


class FixtureMessage(SimpleNamespace):
    def model_dump(self, **_):
        return json.loads(json.dumps(vars(self), default=lambda obj: vars(obj)))

    def to_dict(self):
        return self.model_dump()


def contract(question=BARE, quote="greater than 70", kind="predicate"):
    start = question.index(quote)
    return {"requirements": [{"id": "c1", "kind": kind, "quote": quote,
                              "span": [start, start + len(quote)], "expected": "strictly greater than 70"}], "unknowns": []}


def audit(verdict="FAIL", old=EXPRESSION, new="r['valueQuantity']['value'] > 70"):
    return {"checks": [{"requirement_id": "c1", "verdict": verdict, "reason": "The code includes the boundary value"}],
            "repair": {"requirement_id": "c1", "old_expression": old, "new_expression": new,
                       "reason": "greater than excludes equality"} if verdict == "FAIL" else None}


def final_call(code=CODE, identifier="primary-final"):
    return call("execute_python_code", identifier, code=code, mode="final")


class QueryContractIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.store = deepcopy(STORE)
        self.requests = []
        self.patches = [patch("agent.base_agent.setup_api_keys"),
            patch.object(cache, "CACHE_ENABLED", False),
            patch("agent.single_agent.get_starting_packet", return_value=deepcopy(PACKET)),
            patch("tools.resource_tools.get_fhir_client", return_value=Mock(
                search_with_pagination=Mock(side_effect=lambda *_a, **_kw: deepcopy(self.store)))),
            patch("tools.field_check.fetch_from_store", side_effect=lambda refs: {
                record["id"]: deepcopy(record) for record in self.store
                if any(identifier == record["id"] for _, identifier in refs)})]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_agent(self, responses, question=QUESTION, budget=200_000, tools=30):
        pending = iter(responses)
        def llm(**kwargs):
            self.requests.append(deepcopy(kwargs))
            item = next(pending)
            if isinstance(item, tuple):
                return item
            item = FixtureMessage(**vars(item))
            usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
                     "llm_calls": 1, "cost": 0, "per_call": [{"prompt": 100, "completion": 50, "usage_available": True}]}
            return item, None, usage
        with patch("agent.base_agent.safe_llm_call", side_effect=llm):
            agent = QueryContractAgent("test-model")
            agent.set_budget(max_total_tokens=budget, max_tool_calls=tools)
            output = agent.run(question)
        return agent, output

    def test_complete_real_tool_pipeline_repairs_threshold(self):
        agent, output = self.run_agent([reply(json.dumps(contract())), fetch(), final_call(), reply(json.dumps(audit()))])
        self.assertIsNone(output["error"], output)
        self.assertEqual(output["answer"], 1)
        revision = output["revision"]
        self.assertEqual(revision["draft"]["answer"], 2)
        self.assertTrue(revision["revised"])
        self.assertEqual(revision["release_policy"], "source_checked_targeted_local_constraint")
        self.assertEqual(revision["targeted_proof"]["source_witnesses"][0]["valueQuantity"]["value"], 70)
        self.assertEqual(agent.tool_calls_made, 6)  # retrieve, primary final/check, replay, patched final/check
        self.assertEqual(agent.total_usage["llm_calls"], 4)  # no full-context solver synthesis
        self.assertEqual(output["tokens"]["total"], 600)

    def test_initial_contract_is_draft_blind_and_frozen_before_code_reveal(self):
        _, output = self.run_agent([reply(json.dumps(contract())), fetch(), final_call(), reply(json.dumps(audit("PASS")))])
        initial = json.dumps(self.requests[0]["messages"], default=lambda obj: vars(obj))
        self.assertIn(BARE, initial)
        self.assertNotIn(EXPRESSION, initial)
        self.assertNotIn("Observation/o1", initial)
        final_audit = json.dumps(self.requests[-1]["messages"], default=lambda obj: vars(obj))
        self.assertIn(EXPRESSION, final_audit)
        self.assertNotIn('"candidate_answer"', final_audit)
        from agent.query_contract_agent import fingerprint
        self.assertEqual(output["revision"]["contract_sha256"], fingerprint(output["revision"]["contract"]))
        self.assertEqual(output["answer"], 2)
        from utils.core_utils import process_trace
        trace = process_trace(output["trace"])
        self.assertTrue(any(item.get("phase") == "solver_boundary" and item.get("model_call") is False for item in trace))

    def test_static_audit_is_one_pass_with_no_tools_or_probe_budget(self):
        agent, output = self.run_agent([reply(json.dumps(contract())), fetch(), final_call(),
                                       reply(json.dumps(audit("PASS")))])
        audit_requests = [r for r in self.requests if r['messages'][0]['content'].startswith("Check the solver's actual")]
        self.assertEqual(len(audit_requests), 1)
        self.assertIsNone(audit_requests[0]['tools'])
        self.assertIn("one static code-audit pass with no tools", audit_requests[0]['messages'][0]['content'])
        self.assertEqual(output['revision']['phases'][2]['tool_cap'], 0)
        self.assertEqual(output['revision']['phases'][2]['tool_calls'], 0)
        self.assertEqual(agent.tool_calls_made, 3)

    def test_static_auditor_requested_probe_is_never_executed_or_a_veto(self):
        probe = call("execute_python_code", "forbidden-audit-probe", code="answer=999", mode="inspect")
        agent, output = self.run_agent([reply(json.dumps(contract())), fetch(), final_call(), probe])
        self.assertEqual(output['answer'], 2)
        self.assertFalse(output['revision']['targeted_revised'])
        self.assertIn("requested tools", output['revision']['audit_error'])
        self.assertEqual(agent.tool_calls_made, 3)
        self.assertEqual(output['revision']['phases'][2]['tool_calls'], 0)

    def test_early_unexecuted_final_prose_receives_execution_feedback(self):
        early = reply(json.dumps({'answer': 999, 'evidence': [], 'limitation': None}))
        agent, output = self.run_agent([reply(json.dumps(contract())), early, fetch(), final_call(),
                                       reply(json.dumps(audit('PASS')))])
        self.assertEqual(output['answer'], 2)
        self.assertEqual(agent.total_usage['llm_calls'], 5)
        feedback = [m.get('content') for m in output['trace'] if isinstance(m, dict) and m.get('role')=='user'
                    and 'No valid final computation is available' in (m.get('content') or '')]
        self.assertEqual(len(feedback), 1)
        self.assertIn('mode=final', feedback[0])
        self.assertEqual(output['revision']['draft']['answer'], 2)

    def test_unknown_and_format_failure_preserve_valid_primary(self):
        for tail in [reply(json.dumps(audit("UNKNOWN"))), reply("Looks suspicious."), (None, "provider unavailable", None)]:
            with self.subTest(tail=tail):
                _, output = self.run_agent([reply(json.dumps(contract())), fetch(), final_call(), tail])
                self.assertEqual(output["answer"], 2)
                self.assertIsNone(output["error"])
                self.assertFalse(output["revision"]["revised"])

    def test_bad_planner_format_skips_audit_without_veto(self):
        _, output = self.run_agent([reply("No valid contract"), fetch(), final_call()])
        self.assertEqual(output["answer"], 2)
        self.assertIn("requirements", output["revision"]["audit_error"])
        self.assertEqual(output["tokens"]["llm_calls"], 3)

    def test_invalid_anchor_or_nonlocal_change_cannot_replace(self):
        for proposed in [audit(new="r['valueQuantity']['value'] > 99"),
                         audit(new="r['valueQuantity']['value'] > 70 and r['id'] != 'o2'")]:
            _, output = self.run_agent([reply(json.dumps(contract())), fetch(), final_call(), reply(json.dumps(proposed))])
            self.assertEqual(output["answer"], 2)
            self.assertEqual(output["revision"]["targeted_proof"]["status"], "not_run")
        invalid = contract()
        invalid["requirements"][0]["quote"] = "wrong quote"
        _, output = self.run_agent([reply(json.dumps(invalid)), fetch(), final_call(), reply(json.dumps(audit()))])
        self.assertEqual(output["answer"], 2)
        self.assertEqual(output["revision"]["targeted_proof"]["reason"], "repair lacks a valid anchored FAIL")

    def test_source_failure_of_alternate_preserves_primary(self):
        from agent.query_contract_agent import check_fields
        with patch("agent.query_contract_agent.check_fields", side_effect=RuntimeError("source unavailable")):
            _, output = self.run_agent([reply(json.dumps(contract())), fetch(), final_call(), reply(json.dumps(audit()))])
        self.assertEqual(output["answer"], 2)
        self.assertEqual(output["revision"]["targeted_proof"]["status"], "rejected")
        self.assertTrue(output["revision"]["targeted_proof"]["issues"])

    def test_no_source_budget_cannot_release_alternate(self):
        _, output = self.run_agent([reply(json.dumps(contract())), fetch(), final_call(), reply(json.dumps(audit()))], tools=5)
        self.assertEqual(output["answer"], 2)
        self.assertIn("source check", output["revision"]["targeted_proof"]["reason"])
        self.assertEqual(output["budget"]["tool_calls_run"], 5)

    def test_missing_final_never_releases_copied_prose(self):
        # Repeated unexecuted prose cannot satisfy the team's explicit-final gate.
        responses = [reply(json.dumps(contract())), fetch()] + [reply('{"answer":2,"evidence":[],"limitation":null}')] * 20
        _, output = self.run_agent(responses)
        self.assertIsNone(output["answer"])
        self.assertEqual(output["revision"]["release_policy"], "primary_failed_execution_or_source_contract")

    def test_same_batch_final_then_retrieval_invalidates_readiness(self):
        batch = final_call()
        batch.tool_calls += fetch("later-fetch").tool_calls
        _, output = self.run_agent([reply(json.dumps(contract())), fetch(), batch, final_call(identifier="fresh-final"),
                                   reply(json.dumps(audit("PASS")))])
        self.assertEqual(output["answer"], 2)
        solver_requests = [r for r in self.requests if "You answer questions" in r["messages"][0]["content"]]
        self.assertEqual(len(solver_requests), 3)  # fetch, mixed batch, required fresh final
        finals = [t for t in output["tool_calls"] if t["name"] == "execute_python_code"]
        self.assertEqual(len(finals), 2)

    def test_same_batch_final_then_failed_python_cannot_reuse_final(self):
        batch = final_call()
        batch.tool_calls += call("execute_python_code", "broken", code="raise ValueError('broken')", mode="final").tool_calls
        _, output = self.run_agent([reply(json.dumps(contract())), fetch(), batch, final_call(identifier="fresh-final"),
                                   reply(json.dumps(audit("PASS")))])
        self.assertEqual(output["answer"], 2)
        self.assertEqual(output["tokens"]["llm_calls"], 5)

    def test_inspection_does_not_replace_authoritative_final(self):
        batch = final_call()
        batch.tool_calls += call("execute_python_code", "diagnostic", code="answer={'count':2}", mode="inspect").tool_calls
        _, output = self.run_agent([reply(json.dumps(contract())), fetch(), batch, reply(json.dumps(audit("PASS")))])
        self.assertEqual(output["answer"], 2)
        self.assertEqual(output["tokens"]["llm_calls"], 4)

    def test_primary_source_error_gets_one_local_correction(self):
        bad = CODE + "\nevidence[0]['value'] = 999\n"
        _, output = self.run_agent([reply(json.dumps(contract())), fetch(), final_call(bad), final_call(), reply(json.dumps(audit("PASS")))])
        self.assertEqual(output["answer"], 2)
        self.assertTrue(output["revision"]["primary_revision"]["revised"])
        self.assertEqual(output["tokens"]["llm_calls"], 5)
        self.assertTrue(output["revision"]["solver_source_revised"])
        self.assertFalse(output["revision"]["targeted_revised"])
        self.assertTrue(output["revision"]["revised"])
        self.assertEqual(output["revision"]["draft"]["evidence"][0]["value"], 999)
        self.assertEqual(output["revision"]["final"]["evidence"][0]["value"], 70)

    def test_multiple_threshold_clauses_do_not_replace_correct_primary(self):
        self.store = [observation("o1", 69), observation("o2", 71)]
        bare = "How many weights are less than 70 or greater than 70?"
        question = QUESTION.replace(BARE, bare)
        code = CODE.replace(EXPRESSION, "r['valueQuantity']['value'] < 70 or r['valueQuantity']['value'] > 70")
        _, output = self.run_agent([reply(json.dumps(contract(bare))), fetch(), final_call(code),
            reply(json.dumps(audit(old="r['valueQuantity']['value'] < 70", new="r['valueQuantity']['value'] > 70")))], question=question)
        self.assertEqual(output["answer"], 2)
        self.assertFalse(output["revision"]["targeted_revised"])
        self.assertIn("clause-to-expression", output["revision"]["targeted_proof"]["reason"])

    def test_undated_owner_valid_source_cannot_certify_zero(self):
        self.store = [observation("o1", 71)]
        self.store[0].pop("effectiveDateTime")
        self.store[0].pop("issued")
        _, output = self.run_agent([reply(json.dumps(contract())), fetch(), final_call(), final_call(identifier="second-final")])
        self.assertIsNone(output["answer"])
        revision = output["revision"]
        self.assertTrue(any("source_scope" in str(issue) for issue in revision["final_check_issues"]))
        self.assertEqual(revision["primary_revision"]["draft"]["answer"], 0)
        self.assertIsNone(revision["primary_revision"]["final"]["limitation"])
        checks = [message for message in output["trace"] if isinstance(message, dict) and message.get("name") == "check_fields"]
        self.assertTrue(all("Observation/o1" in message["content"] for message in checks))
        self.assertEqual(revision["final"]["answer"], None)
        # Even before source failure, exclusion was retained in the actual model observation.
        retrieval = next(message for message in output["trace"] if isinstance(message, dict)
                         and message.get("name") == "get_resources_by_patient_fhir_id")
        self.assertEqual(json.loads(retrieval["content"])["excluded"], {"unknown_time": 1})

    def test_unresolved_source_cannot_certify_positive_count_or_aggregates(self):
        self.store = [observation("dated", 71), observation("undated", 72)]
        self.store[1].pop("effectiveDateTime")
        self.store[1].pop("issued")
        for operation, expression in [
                ("count", "len(rows)"),
                ("mean", "sum(r['valueQuantity']['value'] for r in rows) / len(rows)"),
                ("minimum", "min(r['valueQuantity']['value'] for r in rows)"),
                ("maximum", "max(r['valueQuantity']['value'] for r in rows)"),
                ("last", "sorted(rows, key=lambda r: r['effectiveDateTime'])[-1]['valueQuantity']['value']")]:
            with self.subTest(operation=operation):
                code = CODE.replace("answer = len(rows)", "answer = " + expression)
                _, output = self.run_agent([reply(json.dumps(contract())), fetch(), final_call(code),
                                           final_call(code, identifier="second-final")])
                self.assertIsNone(output["answer"], output)
                self.assertGreater(output["revision"]["draft"]["answer"], 0)
                self.assertTrue(any("source_scope" in str(issue) for issue in output["revision"]["final_check_issues"]))
                self.assertEqual(output["revision"]["complete_types"], [])
                self.assertEqual(output["revision"]["unresolved_scope"], {"Observation/undated": "unknown_time"})

    def test_condition_then_encounter_recovers_original_record_scope(self):
        condition = {"resourceType": "Condition", "id": "c1", "subject": {"reference": f"Patient/{PATIENT}"},
                     "encounter": {"reference": "Encounter/e1"}, "code": {"text": "Synthetic condition"}}
        encounter = {"resourceType": "Encounter", "id": "e1", "subject": {"reference": f"Patient/{PATIENT}"},
                     "period": {"start": "2150-01-02T08:00:00", "end": "2150-01-03T08:00:00"}}
        self.store = [condition, encounter]
        client = Mock(search_with_pagination=Mock(side_effect=lambda query: deepcopy(
            [record for record in self.store if record["resourceType"] == query.split("?")[0]])))
        code = """rows = retrieved_resources['Condition']
answer = len(rows)
evidence = [{'resource_id': 'Condition/' + r['id'], 'field_path': 'code.text', 'value': r['code']['text']} for r in rows]
contributing_ids = ['Condition/' + r['id'] for r in rows] + ['Encounter/e1']
operation = 'count condition records'
limitation = None
"""
        condition_fetch = call("get_resources_by_patient_fhir_id", "conditions", resource_type="Condition", patient_fhir_id=PATIENT)
        encounter_fetch = call("get_resources_by_patient_fhir_id", "encounters", resource_type="Encounter", patient_fhir_id=PATIENT)
        with patch("tools.resource_tools.get_fhir_client", return_value=client):
            _, output = self.run_agent([reply(json.dumps(contract())), condition_fetch, encounter_fetch,
                final_call(code), reply(json.dumps(audit("UNKNOWN")))])
        self.assertEqual(output["answer"], 1, output)
        self.assertEqual(output["revision"]["final_check_issues"], [])
        ledger = output["revision"]["retrieval_ledger"]
        self.assertEqual(ledger[0]["unresolved_scope"], {"Condition/c1": "unknown_time"})
        self.assertIsNone(ledger[0]["complete_type"])
        self.assertEqual(ledger[1]["unresolved_scope"], {})
        self.assertEqual(ledger[1]["recovered_scope_ids"], ["Condition/c1"])
        self.assertEqual(output["retrieved_fhir_resources"]["Condition"], [condition])

    def test_medication_closure_before_held_roots_blocks_absence_and_recovers(self):
        request = {"resourceType": "MedicationRequest", "id": "mr1", "subject": {"reference": f"Patient/{PATIENT}"},
                   "authoredOn": "2150-01-02T08:00:00", "dispenseRequest": {"validityPeriod": {"start": "2150-01-02T08:00:00"}},
                   "medicationReference": {"reference": "Medication/m1"}}
        medication = {"resourceType": "Medication", "id": "m1", "code": {"text": "Synthetic medication"}}
        self.store = [request, medication]
        client = Mock(search_with_pagination=Mock(side_effect=lambda query: deepcopy(
            [record for record in self.store if record["resourceType"] == query.split("?")[0]])))
        med_fetch = call("get_resources_by_patient_fhir_id", "medication", resource_type="Medication", patient_fhir_id=PATIENT)
        root_fetch = call("get_resources_by_patient_fhir_id", "requests", resource_type="MedicationRequest", patient_fhir_id=PATIENT)
        code = """rows = retrieved_resources['Medication']
answer = len(rows)
evidence = [{'resource_id': 'Medication/' + r['id'], 'field_path': 'code.text', 'value': r['code']['text']} for r in rows]
contributing_ids = ['Medication/' + r['id'] for r in rows]
operation = 'count held patient-linked medications'
limitation = None
"""
        with patch("tools.resource_tools.get_fhir_client", return_value=client):
            _, absent = self.run_agent([reply(json.dumps(contract())), med_fetch, final_call(code),
                                       final_call(code, identifier="second-final")])
            _, recovered = self.run_agent([reply(json.dumps(contract())), med_fetch, root_fetch, final_call(code),
                                           reply(json.dumps(audit("UNKNOWN")))])
        self.assertIsNone(absent["answer"])
        self.assertEqual(absent["revision"]["unresolved_scope"], {"Medication/m1": "unlinked_metadata"})
        self.assertEqual(absent["revision"]["complete_types"], [])
        self.assertEqual(recovered["answer"], 1, recovered)
        self.assertEqual(recovered["revision"]["final_check_issues"], [])
        self.assertEqual(recovered["revision"]["complete_types"], ["Medication", "MedicationRequest"])
        self.assertEqual(recovered["revision"]["retrieval_ledger"][1]["recovered_scope_ids"], ["Medication/m1"])
        self.assertEqual(recovered["retrieved_fhir_resources"]["Medication"], [medication])

    def test_compensated_count_cannot_be_released_as_local_predicate_repair(self):
        code = CODE.replace("answer = len(rows)", "answer = len(rows) - 1")
        _, output = self.run_agent([reply(json.dumps(contract())), fetch(), final_call(code), reply(json.dumps(audit()))])
        self.assertEqual(output["answer"], 1)
        self.assertFalse(output["revision"]["targeted_revised"])
        self.assertIn("direct count", output["revision"]["targeted_proof"]["reason"])

    def test_incomplete_starting_packet_never_establishes_targeted_proof(self):
        packet = {"resources": {"Observation": deepcopy(self.store)}, "text": "Partial packet"}
        with patch("agent.single_agent.get_starting_packet", return_value=packet):
            _, output = self.run_agent([reply(json.dumps(contract())), final_call(), reply(json.dumps(audit()))])
        self.assertEqual(output["answer"], 2)
        self.assertEqual(output["revision"]["targeted_proof"]["status"], "unsupported")

    def test_budget_signals_visible_and_source_slots_reserved(self):
        _, output = self.run_agent([reply(json.dumps(contract())), fetch(), final_call(), reply(json.dumps(audit("PASS")))])
        solver_requests = [r for r in self.requests if "You answer questions" in r["messages"][0]["content"]]
        self.assertTrue(all("Solver allocation remaining" in r["messages"][-1]["content"] for r in solver_requests))
        self.assertEqual(output["revision"]["phases"][1]["tool_cap"], 27)
        self.assertEqual(output["revision"]["phases"][1]["tool_calls"], 3)

    def test_unknown_provider_usage_fails_global_accounting(self):
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                 "per_call": [{"usage_available": False}]}
        _, output = self.run_agent([(reply(json.dumps(contract())), None, usage)])
        self.assertIsNone(output["answer"])
        self.assertIn("Aggregate usage", output["limitation"])

    def test_genuine_executed_empty_is_not_vetoed(self):
        self.store = []
        code = CODE.replace(">= 70", "> 70")
        _, output = self.run_agent([reply(json.dumps(contract())), fetch(), final_call(code), reply(json.dumps(audit("UNKNOWN")))])
        self.assertEqual(output["answer"], 0)
        self.assertEqual(output["evidence"], [])

    def test_unchanged_contributor_set_is_not_a_counterexample(self):
        code = CODE + "\ncontributing_ids = ['Observation/' + r['id'] for r in retrieved_resources['Observation']]\n"
        _, output = self.run_agent([reply(json.dumps(contract())), fetch(), final_call(code), reply(json.dumps(audit()))])
        self.assertEqual(output["answer"], 2)
        self.assertIn("direct count", output["revision"]["targeted_proof"]["reason"])

    def test_boundary_row_in_union_cannot_explain_unrelated_changed_ids(self):
        self.store.append(observation("o3", 69))
        code = CODE + "\ncontributing_ids += ['Observation/o3'] * (len(rows) // 2)\n"
        _, output = self.run_agent([reply(json.dumps(contract())), fetch(), final_call(code), reply(json.dumps(audit()))])
        self.assertEqual(output["answer"], 2)
        self.assertIn("direct count", output["revision"]["targeted_proof"]["reason"])

    def test_ordinal_source_trace_establishes_exact_old_new_rank(self):
        self.store[1]["effectiveDateTime"] = "2150-01-03T08:00:00"
        bare = "What was the last measured weight?"
        question = QUESTION.replace(BARE, bare)
        code = """ordered = sorted(retrieved_resources['Observation'], key=lambda r: r['effectiveDateTime'])
selected = ordered[0]
answer = selected['valueQuantity']['value']
evidence = [{'resource_id': 'Observation/' + selected['id'], 'field_path': 'valueQuantity.value', 'value': answer}]
contributing_ids = ['Observation/' + selected['id']]
operation = 'select one ordered measurement'
limitation = None
"""
        _, output = self.run_agent([reply(json.dumps(contract(bare, "last measured", "ordinal"))), fetch(), final_call(code),
                                   reply(json.dumps(audit(old="ordered[0]", new="ordered[-1]")))], question=question)
        self.assertEqual(output["answer"], 71)
        self.assertEqual(output["revision"]["targeted_proof"]["ordered_ids"], ["Observation/o1", "Observation/o2"])
        self.assertEqual(output["revision"]["final_check_issues"], [])

    def test_identical_timestamp_cannot_establish_material_ordinal_error(self):
        bare = "What was the last measured weight?"
        question = QUESTION.replace(BARE, bare)
        code = """ordered = sorted(retrieved_resources['Observation'], key=lambda r: r['effectiveDateTime'])
selected = ordered[0]
answer = selected['valueQuantity']['value']
evidence = [{'resource_id': 'Observation/' + selected['id'], 'field_path': 'valueQuantity.value', 'value': answer}]
contributing_ids = ['Observation/' + selected['id']]
operation = 'select one ordered measurement'
limitation = None
"""
        _, output = self.run_agent([reply(json.dumps(contract(bare, "last measured", "ordinal"))), fetch(), final_call(code),
                                   reply(json.dumps(audit(old="ordered[0]", new="ordered[-1]")))], question=question)
        self.assertEqual(output["answer"], 70)
        self.assertIn("exact requested rank", output["revision"]["targeted_proof"]["reason"])


class QueryContractUnitTests(unittest.TestCase):
    def test_final_static_protocol_has_exact_fixed_aggregate_caps(self):
        self.assertEqual(PHASE_CAPS, {'contract': (10000,0), 'solver': (175000,27),
                                     'audit': (15000,0), 'targeted': (0,3)})
        self.assertEqual(sum(cap[0] for cap in PHASE_CAPS.values()), 200000)
        self.assertEqual(sum(cap[1] for cap in PHASE_CAPS.values()), 30)

    def test_unique_quote_host_normalizes_bad_character_count(self):
        proposed = contract()
        proposed["requirements"][0]["span"] = [999, 1000]
        parsed, error = parse_contract(json.dumps(proposed), BARE)
        self.assertIsNone(error)
        item = parsed["requirements"][0]
        self.assertTrue(item["anchor_valid"])
        self.assertTrue(item["span_normalized"])
        self.assertEqual(item["span_supplied"], [999, 1000])
        self.assertEqual(BARE[item["span"][0]:item["span"][1]], "greater than 70")

    def test_repeated_or_missing_exact_quotes_do_not_anchor(self):
        for question, quote in [("Was first measured weight above first measured glucose?", "first measured"),
                                (BARE, "not actually present")]:
            proposed = {"requirements": [{"id": "c1", "kind": "ordinal", "quote": quote,
                                           "span": [0, len(quote)], "expected": "last"}]}
            parsed, _ = parse_contract(json.dumps(proposed), question)
            self.assertFalse(parsed["requirements"][0]["anchor_valid"])
            self.assertIsNone(parsed["requirements"][0]["span"])

    def test_ordinal_patch_requires_effective_datetime_sort(self):
        question = "What was the last measured weight?"
        parsed, error = parse_contract(json.dumps(contract(question, "last measured", "ordinal")), question)
        self.assertIsNone(error)
        code = "ordered = sorted(rows, key=lambda r: r['effectiveDateTime'])\nanswer = ordered[0]['valueQuantity']['value']"
        proposed = audit(old="ordered[0]", new="ordered[-1]")
        result, error = patched_program(code, proposed, parsed, question)
        self.assertIsNone(error)
        self.assertIn("ordered[-1]", result["code"])
        result, error = patched_program(code.replace("effectiveDateTime", "issued"), proposed, parsed, question)
        self.assertIsNone(result)
        self.assertIn("effectiveDateTime", error)

    def test_alias_or_multiple_occurrences_are_not_claimed_proved(self):
        parsed, _ = parse_contract(json.dumps(contract()), BARE)
        result, error = patched_program(CODE + "\nextra = " + EXPRESSION, audit(), parsed, BARE)
        self.assertIsNone(result)
        self.assertIn("exactly once", error)
        result, error = patched_program("answer = value >= 70", audit(old="value >= 70", new="value > 70"), parsed, BARE)
        self.assertIsNone(result)

    def test_negative_synonym_or_multiple_numeric_code_clauses_are_unsupported(self):
        for question in ["How many weights are not greater than 70?",
                         "How many weights are below 70 or greater than 70?"]:
            parsed, _ = parse_contract(json.dumps(contract(question)), question)
            result, error = patched_program(CODE, audit(), parsed, question)
            self.assertIsNone(result)
            self.assertIn("clause binding", error)
        parsed, _ = parse_contract(json.dumps(contract()), BARE)
        result, error = patched_program(CODE + "\nnormal_height = height >= 70", audit(), parsed, BARE)
        self.assertIsNone(result)
        self.assertIn("program comparisons", error)

    def test_phase_output_reservation_respects_local_and_global_caps(self):
        with patch("agent.base_agent.setup_api_keys"):
            agent = QueryContractAgent("test-model")
        agent.set_budget(max_total_tokens=200_000, max_tool_calls=30)
        phase = PhaseHost(agent, "contract")
        seen = []
        with patch("agent.base_agent.safe_llm_call", side_effect=lambda **kw: (seen.append(kw) or reply("{}"), None, None)):
            phase._llm_call([{"role": "user", "content": "short prompt"}])
        self.assertLessEqual(seen[0]["max_completion_tokens"], PHASE_CAPS["contract"][0])
        self.assertEqual(agent.max_total_tokens, 200_000)

    def test_phase_refusal_restores_reserve_for_later_phases(self):
        with patch("agent.base_agent.setup_api_keys"):
            agent = QueryContractAgent("test-model")
        agent.set_budget(max_total_tokens=200_000, max_tool_calls=30)
        phase = PhaseHost(agent, "contract")
        phase.start_tokens = 0
        agent.total_usage["total_tokens"] = 9999
        with patch("agent.base_agent.safe_llm_call") as provider:
            _, error, _ = phase._llm_call([{"role": "user", "content": "next prompt"}])
        self.assertIn("Budget exceeded", error)
        provider.assert_not_called()
        self.assertEqual(agent.max_total_tokens, 200_000)
        self.assertIsNone(agent.last_llm_error)
        self.assertIsNone(agent.budget_stop)
        next_phase = PhaseHost(agent, "solver")
        with patch("agent.base_agent.safe_llm_call", return_value=(reply("{}"), None, None)) as provider:
            next_phase._llm_call([{"role": "user", "content": "new phase"}])
        self.assertGreater(provider.call_args.kwargs["max_completion_tokens"], 170_000)

    def test_provider_overshoot_of_local_phase_is_fatal_even_below_global_cap(self):
        with patch("agent.base_agent.setup_api_keys"):
            agent = QueryContractAgent("test-model")
        agent.set_budget(max_total_tokens=200_000, max_tool_calls=30)
        phase = PhaseHost(agent, "contract")
        usage = {"prompt_tokens": 100, "completion_tokens": 9901, "total_tokens": 10001,
                 "per_call": [{"usage_available": True}]}
        with patch("agent.base_agent.safe_llm_call", return_value=(reply("{}"), None, usage)):
            _, error, _ = phase._llm_call([{"role": "user", "content": "short prompt"}])
        self.assertTrue(phase.fatal)
        self.assertIn("Provider accounting failure", error)
        self.assertEqual(agent.max_total_tokens, 200_000)
        self.assertIsNotNone(agent.budget_stop)

    def test_changed_source_membership_guard_even_when_counts_match_contributors(self):
        with patch("agent.base_agent.setup_api_keys"):
            agent = QueryContractAgent("test-model")
        primary = {"answer": 3, "evidence": [], "operation": "count", "limitation": None,
                   "contributing_ids": ["Observation/o1", "Observation/o2", "Observation/o3"]}
        alternate = {**primary, "answer": 1, "contributing_ids": ["Observation/o2"]}
        workspace = {"Observation": deepcopy(STORE) + [observation("o3", 69)]}
        with patch.object(agent, "_execute", side_effect=[deepcopy(primary), deepcopy(alternate)]):
            result, proof = agent._targeted({"kind": "predicate", "code": "pass", "threshold": 70,
                                            "old_operator": "GtE", "new_operator": "Gt"},
                primary, {"code": "pass"}, workspace, {"question": BARE}, {"Observation"}, PhaseHost(agent, "targeted"))
        self.assertIsNone(result)
        self.assertIn("actual old/new predicate", proof["reason"])

    def test_source_slots_are_counted_inside_solver_and_global_budget(self):
        with patch("agent.base_agent.setup_api_keys"):
            agent = QueryContractAgent("test-model")
            agent.set_budget(max_total_tokens=200_000, max_tool_calls=30)
            phase = PhaseHost(agent, "solver")
            solver = LedgerSolver(agent, phase)
        for _ in range(25):
            self.assertTrue(solver._solver_take_tool_call())
        self.assertFalse(solver._solver_take_tool_call())
        solver._source_check_active = True
        self.assertTrue(solver._solver_take_tool_call())
        self.assertTrue(solver._solver_take_tool_call())
        self.assertFalse(solver._solver_take_tool_call())
        self.assertEqual(agent.tool_calls_made, 27)
        self.assertEqual(phase.report()["tool_calls"], 27)

    def test_denied_later_retrieval_clears_final_readiness(self):
        with patch("agent.base_agent.setup_api_keys"):
            agent = QueryContractAgent("test-model")
            solver = LedgerSolver(agent, PhaseHost(agent, "solver"))
        solver.final_computation = {"answer": 2}
        solver._tool_budget_message(fetch().tool_calls[0])
        self.assertIsNone(solver.final_computation)

    def test_rank_binding_rejects_mutation_deletion_and_shadowing(self):
        question = "What was the last measured weight?"
        parsed, _ = parse_contract(json.dumps(contract(question, "last measured", "ordinal")), question)
        prefix = "ordered = sorted(rows, key=lambda r: r['effectiveDateTime'])\n"
        suffix = "answer = ordered[0]['valueQuantity']['value']\n"
        proposed = audit(old="ordered[0]", new="ordered[-1]")
        for middle in ["ordered += rows\n", "del ordered\n", "for ordered in rows: pass\n",
                       "def f(ordered): pass\n", "ordered.reverse()\n", "ordered[0] = rows[1]\n",
                       "ordered[0]['effectiveDateTime'] = '2150-01-09'\n"]:
            with self.subTest(middle=middle):
                result, error = patched_program(prefix + middle + suffix, proposed, parsed, question)
                self.assertIsNone(result)

    def test_multiple_measurement_ranks_cannot_bind_unique_code_index(self):
        question = "What is the first measured weight minus the last measured height?"
        parsed, _ = parse_contract(json.dumps(contract(question, "last measured height", "ordinal")), question)
        code = "ordered = sorted(weights, key=lambda r: r['effectiveDateTime'])\nanswer = ordered[0]['valueQuantity']['value']"
        result, error = patched_program(code, audit(old="ordered[0]", new="ordered[-1]"), parsed, question)
        self.assertIsNone(result)
        self.assertIn("clause-to-expression", error)
        question = "What is the first weight minus the last measured height?"
        parsed, _ = parse_contract(json.dumps(contract(question, "last measured height", "ordinal")), question)
        result, error = patched_program(code, audit(old="ordered[0]", new="ordered[-1]"), parsed, question)
        self.assertIsNone(result)

    def test_scope_completeness_recomputed_after_new_unknown_record(self):
        from agent.query_contract_agent import ScopedLedgerToolbox
        with patch("agent.base_agent.setup_api_keys"):
            agent = QueryContractAgent("test-model")
            solver = LedgerSolver(agent, PhaseHost(agent, "solver"))
        solver._scoped_box = ScopedLedgerToolbox((), {}, patient_fhir_id=PATIENT, current_time=NOW)
        solver.retrieval_ledger = [{"tool": "get_resources_by_patient_fhir_id", "success": True,
                                    "arguments": {"resource_type": "Observation"}, "complete_type": "Observation"}]
        self.assertEqual(solver.complete_types(), {"Observation"})
        solver._scoped_box.scope_failures["Observation/new"] = "unknown_time"
        self.assertEqual(solver.complete_types(), set())

    def test_changed_non_observation_dependencies_are_rejected(self):
        with patch("agent.base_agent.setup_api_keys"):
            agent = QueryContractAgent("test-model")
        agent.set_budget(max_tool_calls=30)
        primary = {"answer": 2, "evidence": [], "operation": "count", "limitation": None,
                   "contributing_ids": ["Observation/o1", "Observation/o2", "Patient/" + PATIENT]}
        alternate = {**primary, "answer": 1, "contributing_ids": ["Observation/o2"]}
        phase = PhaseHost(agent, "targeted")
        with patch.object(agent, "_execute", side_effect=[deepcopy(primary), deepcopy(alternate)]):
            result, proof = agent._targeted({"kind": "predicate", "code": "pass"}, primary,
                {"code": "pass"}, {"Observation": deepcopy(STORE)}, {"question": BARE}, {"Observation", "Patient"}, phase)
        self.assertIsNone(result)
        self.assertIn("non-Observation", proof["reason"])


class QueryContractEntryTests(unittest.TestCase):
    def test_additive_worker_uses_common_budget_and_output_wrapper(self):
        from agent.query_contract_agent import run_v3_job
        job = ("synthetic question", "multi_agent_v3", "same-model", False, None, False,
               {"max_total_tokens": 200_000, "max_tool_calls": 30}, None)
        with patch("agent.query_contract_agent.QueryContractAgent") as factory, \
                patch("utils.run_agent_safe", return_value={"agent_answer": 1}) as wrapper:
            result = run_v3_job(job)
        factory.assert_called_once_with("same-model", verbose=False, base_url=None)
        factory.return_value.set_budget.assert_called_once_with(max_total_tokens=200_000, max_tool_calls=30)
        factory.return_value.set_perturbation.assert_called_once_with(None)
        wrapper.assert_called_once_with(factory.return_value, "synthetic question")
        self.assertEqual(result, {"agent_answer": 1})

    def test_legacy_worker_route_is_unchanged(self):
        import run_agent
        job = ("question", "single_agent", "same-model", False, None, True, {}, None)
        with patch("run_agent.run_agent_with_input", return_value="legacy") as worker:
            self.assertEqual(run_agent.dispatch_agent_job(job), "legacy")
        worker.assert_called_once_with(job)

    def test_phase3_team_flag_records_actual_strategy_without_renaming_arm(self):
        from scripts.run_experiments import Arm, runner_argv
        args = SimpleNamespace(team_strategy="multi_agent_v3", model="same-model", input="synthetic.csv",
            output_dir="/private/tmp/v3-output", split_manifest="frozen.json", split="dev",
            num_processes=1, save_interval=1, base_url=None, max_total_tokens=200_000,
            max_tool_calls=30, limit=300, verbose=False)
        argv = runner_argv(Arm("ordinary", "multi_agent"), args)
        self.assertEqual(argv[argv.index("--agent_strategy") + 1], "multi_agent_v3")
        self.assertTrue(argv[argv.index("--output") + 1].endswith("ordinary_multi_agent.json"))
        unchanged = runner_argv(Arm("ordinary", "single_agent"), args)
        self.assertEqual(unchanged[unchanged.index("--agent_strategy") + 1], "single_agent")

    def test_full_benchmark_team_flag_is_additive(self):
        from pathlib import Path
        from scripts.full_benchmark import arm_command
        args = SimpleNamespace(team_strategy="multi_agent_v3", model="same-model", input=Path("synthetic.csv"),
                              output_dir=Path("/private/tmp/v3-full"), max_total_tokens=200_000,
                              max_tool_calls=30, num_processes=1, save_interval=1)
        command = arm_command("multi_agent", args)
        self.assertEqual(command[command.index("--agent_strategy") + 1], "multi_agent_v3")
        self.assertTrue(command[command.index("--output") + 1].endswith("full_multi_agent.json"))


if __name__ == "__main__":
    unittest.main()
