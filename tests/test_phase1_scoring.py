"""Regression checks for deterministic answer/evidence evaluation; no API calls."""
import contextlib
import csv
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd

from utils.answer_scorer import answer_type, parse_reference, score_answer
from utils.evidence_scorer import (
    check_citation, cited_refs, fetch_from_ndjson, fetch_from_store,
    get_cited_resource, parse_evidence, score_evidence, values_match,
)
from utils.metrics_utils import retrieval_precision, retrieval_recall

ROOT = Path(__file__).resolve().parents[1]


class AnswerScoringTests(unittest.TestCase):
    def assert_correct(self, answer, reference, expected=1, question="", template=""):
        result = score_answer(answer, reference, question, template)
        self.assertEqual(result["typed_correct"], expected, result)

    def test_boolean_requires_zero_one_or_bool(self):
        for answer in (2, -1, 0.5, float("nan"), float("inf")):
            with self.subTest(answer=answer):
                self.assert_correct(answer, "[[1]]", 0, template="Has_verb patient")
        self.assert_correct(True, "[[1]]", template="Has_verb patient")
        self.assert_correct(0, "[[0]]", template="Has_verb patient")

    def test_count_is_exact_but_platelet_measurement_uses_tolerance(self):
        self.assert_correct(1000000001, "[[1000000000]]", 0, template="Count the number of visits")
        self.assert_correct(2.001, "[[2]]", 0, question="How many hospital admissions occurred?")
        self.assert_correct(201.5, "[[201]]", question="What was the platelet count?",
                            template="What was the measured {lab_name} value?")
        self.assert_correct("2 kg", "[[2]]", 0, template="Count the number of visits")

    def test_scalar_numeric_answer_does_not_choose_a_matching_component(self):
        self.assert_correct("The component was 80 but the total was 100.", "[[80]]", 0)
        self.assert_correct("12.3 kg", "[[12.3]]", 0, question="What was the PT value?")

    def test_scalar_units_are_converted_and_unknown_claims_rejected(self):
        template = "What was the measured weight?"
        self.assert_correct("176.3698 lb", "[[80]]", template=template)
        self.assert_correct("80 ml", "[[80]]", 0, template=template)
        self.assert_correct("80 bananas", "[[80]]", 0, template=template)
        self.assert_correct("80 and recorded yesterday", "[[80]]", template=template)

    def test_standalone_answer_can_equal_question_threshold(self):
        self.assert_correct("12.3", "[[12.3]]", question="Was the minimum value below 12.3?")
        self.assert_correct("2186-09-15 18:00:00", "[['2186-09-15 18:00:00']]",
                            question="When was the test on 2186-09-15 18:00:00?")

    def test_empty_zero_is_only_a_numeric_answer(self):
        self.assert_correct(0, "[]", question="What is the gender?", expected=0)
        self.assert_correct("0", "[]", question="What is the gender?", expected=0)
        self.assert_correct(0, "[[None]]", template="Is the weight greater?", expected=0)
        self.assert_correct(False, "[[None]]", template="Is the weight greater?", expected=0)
        self.assert_correct(0, "[]", template="Count the number of visits")
        self.assert_correct(None, "[]")
        self.assert_correct("No records were found", "[]")

    def test_negative_statement_cannot_hide_a_conflicting_empty_value(self):
        self.assert_correct("No, the dose was 5 mg.", "[]", 0, question="What was the dose?")
        self.assert_correct("No, it happened on 2186-09-15 18:00.", "[]", 0, question="When?")

    def test_numeric_sets_require_all_rows_without_extra_values(self):
        self.assert_correct("1, 2, 3", "[[1], [2]]", 0)
        self.assert_correct([1, 1], "[[1], [2]]", 0)
        self.assert_correct([1], "[[1], [1]]", 0)
        self.assert_correct([2, 1], "[[1], [2]]")
        self.assert_correct([None, 1], "[[1], [2]]", 0)
        self.assert_correct(["unavailable", 1], "[[1], [2]]", 0)
        self.assert_correct([float("nan"), 1], "[[1], [2]]", 0)

    def test_numeric_set_units_are_checked(self):
        template = "List the monthly average weight"
        self.assert_correct("176.3698 lb, 198.4160 lb", "[[80], [90]]", template=template)
        self.assert_correct("80 ml, 90 ml", "[[80], [90]]", 0, template=template)

    def test_csv_serialized_category_lists_have_no_extra_values(self):
        self.assert_correct('["Heparin", "Aspirin"]', "[['Heparin'], ['Aspirin']]")
        self.assert_correct('["Heparin", "Aspirin", "Insulin"]', "[['Heparin'], ['Aspirin']]", 0)
        self.assert_correct(['male', 'female'], "[['M'], ['F']]")
        self.assert_correct("Heparin and Aspirin", "[['Heparin'], ['Aspirin']]")
        self.assert_correct("Heparin, Aspirin, Insulin", "[['Heparin'], ['Aspirin']]", 0)
        self.assert_correct("Heparin Aspirin", "[['Heparin'], ['Aspirin']]", 0)

    def test_negated_category_is_not_a_match(self):
        self.assert_correct("The patient is not male.", "[['M']]", 0)
        self.assert_correct("male female", "[['M']]", 0)
        self.assert_correct("Heparin, Aspirin", "[['Heparin']]", 0)
        self.assert_correct("The patient's sex is male.", "[['M']]")

    def test_timestamp_sets_use_requested_precision_and_reject_extras(self):
        reference = "[['2186-09-15 18:00:00'], ['2186-09-16 19:00:00']]"
        self.assert_correct(['2186-09-16T19:00:30Z', '2186-09-15T18:00:15Z'], reference)
        self.assert_correct(['2186-09-15', '2186-09-16'], reference, 0)
        self.assert_correct(['2186-09-15', '2186-09-16'], reference, question="What dates?")
        self.assert_correct(['invalid', '2186-09-16'], reference, 0)
        self.assert_correct("2186-09-15 18:00, 2186-09-16 19:00, 2186-09-17 20:00", reference, 0)

    def test_explicit_second_precision(self):
        self.assert_correct("2186-09-15 18:00:59", "[['2186-09-15 18:00:00']]", 0,
                            question="At what second was the measurement recorded?")

    def test_all_released_references_round_trip_without_crashing(self):
        with (ROOT / "final_dataset/questions_answers_sql_fhir.csv").open() as stream:
            for row in csv.DictReader(stream):
                values = parse_reference(row["true_answer"])
                answer = (values[0] if len(values) == 1 else values) if values else None
                with self.subTest(question_id=row["question_id"]):
                    self.assert_correct(answer, row["true_answer"], question=row["question"], template=row["template"])


class EvidenceScoringTests(unittest.TestCase):
    def setUp(self):
        self.observation = {"id": "shared", "resourceType": "Observation", "active": True,
                            "valueQuantity": {"value": 80, "unit": "kg"},
                            "effectiveDateTime": "2186-09-15T18:00:00Z", "nullable": None,
                            "encounter": {"reference": "Encounter/shared"},
                            "code": {"coding": [{"display": "Weight"}, {"display": "Body weight"}]}}
        self.encounter = {"id": "shared", "resourceType": "Encounter", "status": "finished"}
        self.resources = {"Observation/shared": self.observation, "Encounter/shared": self.encounter}

    def citation(self, path, value, ref="Observation/shared", **extra):
        return {"resource_id": ref, "field_path": path, "value": value, **extra}

    def test_type_qualified_identity_and_ambiguous_bare_ids(self):
        self.assertEqual(check_citation(self.citation("valueQuantity.value", 80), self.resources), "valid")
        self.assertEqual(check_citation(self.citation("status", "finished", "Encounter/shared"), self.resources), "valid")
        self.assertEqual(check_citation(self.citation("status", "finished"), self.resources), "bad_path")
        self.assertIsNone(get_cited_resource(self.resources, None, "shared"))
        self.assertEqual(check_citation(self.citation("valueQuantity.value", 80, "shared"), self.resources), "missing_resource")

    def test_bool_does_not_match_numeric_one_and_null_is_exact(self):
        self.assertFalse(values_match(1, True))
        self.assertFalse(values_match(False, 0))
        self.assertTrue(values_match("true", True))
        self.assertEqual(check_citation(self.citation("nullable", None), self.resources), "valid")
        self.assertEqual(check_citation(self.citation("nullable", "None"), self.resources), "value_mismatch")

    def test_paths_lists_units_timestamps_and_encounter_links(self):
        self.assertEqual(check_citation(self.citation("code.coding.display", ["Weight", "Body weight"]), self.resources), "valid")
        self.assertEqual(check_citation(self.citation("code.coding[1].display", "Body weight"), self.resources), "valid")
        self.assertEqual(check_citation(self.citation("code.coding[2].display", "Weight"), self.resources), "bad_path")
        self.assertEqual(check_citation(self.citation("valueQuantity.unit", "ml"), self.resources), "value_mismatch")
        self.assertEqual(check_citation(self.citation("encounter.reference", "Encounter/wrong"), self.resources), "value_mismatch")
        self.assertEqual(check_citation(self.citation("effectiveDateTime", "2186-09-15 18:00:00"), self.resources), "valid")
        self.assertEqual(check_citation(self.citation("effectiveDateTime", "2186-09-15"), self.resources), "value_mismatch")
        self.assertTrue(values_match({"coding": [{"display": "weight"}]}, {"coding": [{"display": "Weight", "code": "x"}]}))
        self.assertFalse(values_match({}, {"value": 80}))

    def test_malformed_type_or_conflicting_type_does_not_crash(self):
        malformed = [self.citation("valueQuantity.value", 80, resource_type=[]),
                     self.citation("valueQuantity.value", 80, resource_type="Encounter"),
                     self.citation("", 80), self.citation("value", 80, "Observation/"),
                     self.citation("value", 80, "Observation/shared?_count=1000"),
                     self.citation("value", 80, "UnknownType/shared")]
        self.assertEqual(cited_refs([malformed]), set())
        result = score_evidence(malformed, self.resources, {"Observation": ["shared"]})
        self.assertEqual(result["evidence_issues"], {"malformed": 6})

    def test_typed_evidence_overlap_penalizes_wrong_resource_type(self):
        result = score_evidence([self.citation("status", "finished", "Encounter/shared")], self.resources,
                                {"Observation": ["shared"]})
        self.assertEqual(result["evidence_precision"], 0)
        self.assertEqual(result["evidence_recall"], 0)
        self.assertEqual(result["evidence_valid"], 1)

    def test_missing_and_malformed_evidence_are_counted(self):
        citations = [self.citation("valueQuantity.value", 80), self.citation("missing", 80),
                     self.citation("valueQuantity.value", 90), self.citation("value", 1, "Observation/absent"), "invalid"]
        result = score_evidence(citations, self.resources, {"Observation": ["shared"]})
        self.assertEqual(result["evidence_cited"], 5)
        self.assertEqual(result["valid_evidence_fraction"], 0.2)
        self.assertEqual(result["evidence_issues"], {"bad_path": 1, "value_mismatch": 1, "missing_resource": 1, "malformed": 1})
        self.assertEqual(parse_evidence("not json"), ["not json"])
        self.assertEqual(parse_evidence(float("nan")), [])

    def test_ndjson_fetch_preserves_both_resources_with_same_id(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "resources.ndjson"
            path.write_text(json.dumps(self.observation, separators=(",", ":\t")) + "\n" + json.dumps(self.encounter) + "\n")
            resources = fetch_from_ndjson({("Observation", "shared"), ("Encounter", "shared")}, tempdir)
        self.assertNotIn("shared", resources)
        self.assertEqual(check_citation(self.citation("valueQuantity.value", 80), resources), "valid")
        self.assertEqual(check_citation(self.citation("status", "finished", "Encounter/shared"), resources), "valid")

    def test_gzip_ndjson_uses_the_root_id_regardless_of_key_order(self):
        nested_first = {"identifier": [{"id": "nested-id"}], **self.observation}
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "observations.ndjson.gz"
            path.write_bytes(gzip.compress((json.dumps(nested_first) + "\n").encode()))
            resources = fetch_from_ndjson({("Observation", "shared")}, tempdir)
        self.assertEqual(check_citation(self.citation("valueQuantity.value", 80), resources), "valid")

    def test_missing_ndjson_configuration_is_an_error(self):
        with tempfile.TemporaryDirectory() as tempdir, self.assertRaises(FileNotFoundError):
            fetch_from_ndjson({("Observation", "shared")}, tempdir)

    def test_store_fetch_preserves_types_without_a_live_store(self):
        client = types.SimpleNamespace(get_resources_by_resource_ids=lambda rtype, ids:
                                      [self.observation if rtype == "Observation" else self.encounter])
        with patch("fhir_client.get_fhir_client", return_value=client):
            resources = fetch_from_store({("Observation", "shared"), ("Encounter", "shared")})
        self.assertNotIn("shared", resources)
        self.assertIs(get_cited_resource(resources, "Observation", "shared"), self.observation)


class EvaluationPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("phase1_evaluation", ROOT / "evaluation_metrics.py")
        cls.evaluation = importlib.util.module_from_spec(spec)
        with patch.object(sys, "argv", ["evaluation_metrics.py", "--input", "unused.json", "--skip_llm_judge"]):
            spec.loader.exec_module(cls.evaluation)

    def test_all_retrieved_types_are_in_precision_denominator(self):
        frame = pd.DataFrame([{"true_fhir_ids": {"Observation": ["shared"]},
                               "agent_fhir_resources": {"Observation": ["shared", "shared"], "Encounter": ["shared"]}}])
        with contextlib.redirect_stdout(io.StringIO()):
            scored = self.evaluation.calculate_retrieval_metrics(frame)
        self.assertEqual(scored.iloc[0]["precision"], 0.5)
        self.assertEqual(scored.iloc[0]["recall"], 1)
        self.assertEqual(retrieval_precision(["a", "a", "b"], ["a"]), 0.5)
        self.assertEqual(retrieval_recall(["a"], ["a", "a", "b"], method="continuous"), 0.5)

    def test_resource_parsing_accepts_csv_json_and_handles_invalid_values(self):
        self.assertEqual(self.evaluation.extract_agent_resources({"agent_fhir_resources": '{"Observation": ["one"]}'}),
                         ["Observation/one"])
        self.assertEqual(self.evaluation.extract_agent_resources({"agent_fhir_resources": {"Observation": None}}), [])
        self.assertEqual(self.evaluation.extract_agent_resources({"agent_fhir_resources": {"Observation": "one"}}),
                         ["Observation/one"])
        self.assertEqual(self.evaluation.extract_agent_resources({"agent_fhir_resources": "not JSON"}), [])

    def test_input_cannot_replace_gold_labels_and_unknown_ids_are_rejected(self):
        ground_truth = pd.DataFrame([{"question_id": "known", "true_answer": "[[2]]", "question": "How many?",
                                     "true_fhir_ids": "{'Encounter': ['gold']}"}])
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "input.json"
            path.write_text(json.dumps([{"question_id": "known", "agent_answer": 1, "true_answer": "[[1]]",
                                         "true_fhir_ids": {"Encounter": ["incorrect"]}}]))
            with patch.object(self.evaluation.args, "input", str(path)), \
                    patch.object(self.evaluation.pd, "read_csv", side_effect=lambda *args, **kwargs: ground_truth.copy()):
                with contextlib.redirect_stdout(io.StringIO()):
                    frame = self.evaluation.load_and_merge_data()
                self.assertEqual(frame.iloc[0]["true_answer"], "[[2]]")
                self.assertEqual(frame.iloc[0]["true_fhir_ids"], {"Encounter": ["gold"]})
                path.write_text(json.dumps([{"question_id": "unknown", "agent_answer": 1}]))
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "absent"):
                    self.evaluation.load_and_merge_data()

    def test_errors_keep_rows_in_accuracy_denominator(self):
        frame = pd.DataFrame([{"agent_answer": None, "true_answer": "[]", "question": "What dose?", "error": "failed"},
                               {"agent_answer": 2, "true_answer": "[[2]]", "question": "How many?", "error": None}])
        with contextlib.redirect_stdout(io.StringIO()):
            scored = self.evaluation.calculate_typed_metrics(frame)
        self.assertEqual(scored["typed_correct"].tolist(), [0, 1])

    def test_judge_reference_uses_answer_nullness_not_missing_resource_annotations(self):
        frame = pd.DataFrame([{"agent_answer": False, "true_answer": "[[0]]", "question": "Has it happened?",
                               "true_fhir_resource_types": "{}"},
                              {"agent_answer": None, "true_answer": "[[None]]", "question": "What dose?",
                               "true_fhir_resource_types": "{Observation}"}])
        with patch.object(self.evaluation, "call_model", return_value="1"), \
                patch.object(self.evaluation, "normalize_null_answers_with_llm", return_value="question answered"), \
                contextlib.redirect_stdout(io.StringIO()):
            scored = self.evaluation.calculate_answer_metrics(frame)
        self.assertEqual(scored["true_answer_extracted"].tolist(), ["[[0]]", "no answer"])

    def test_saved_evaluation_retains_audit_fields(self):
        data = {"question_id": "known", "typed_correct": 1, "typed_parsed": 80,
                "tokens": {"total": 100}, "latency": 1.2, "revision": {"draft_answer": 90},
                "budget": {"stopped": None}, "limitation": None, "tool_calls": [], "error": None}
        with tempfile.TemporaryDirectory() as tempdir, patch.object(self.evaluation.args, "output_dir", tempdir), \
                contextlib.redirect_stdout(io.StringIO()):
            path = self.evaluation.save_results(pd.DataFrame([data]))
            saved = json.loads(Path(path).read_text())[0]
        for field in data:
            self.assertIn(field, saved)

    def test_offline_cli_scores_every_type_without_model_or_store_calls(self):
        predictions = {}
        with (ROOT / "final_dataset/questions_answers_sql_fhir.csv").open() as stream:
            for row in csv.DictReader(stream):
                kind = answer_type(row["true_answer"], row["template"])
                if kind in predictions:
                    continue
                values = parse_reference(row["true_answer"])
                predictions[kind] = {"question_id": row["question_id"],
                                     "agent_answer": values[0] if len(values) == 1 else values or None,
                                     "evidence": [], "agent_fhir_resources": {}}
        self.assertEqual(set(predictions), {"number", "boolean", "empty", "timestamp", "category", "set"})
        with tempfile.TemporaryDirectory() as tempdir:
            source = Path(tempdir) / "predictions.json"
            source.write_text(json.dumps(list(predictions.values())))
            result = subprocess.run([sys.executable, "evaluation_metrics.py", "--input", str(source),
                                     "--output_dir", tempdir, "--skip_llm_judge", "--evidence_source", "ndjson",
                                     "--fhir_dir", tempdir], cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            saved = json.loads((Path(tempdir) / "predictions_eval.json").read_text())
            self.assertEqual(len(saved), 6)
            self.assertTrue(all(row["typed_correct"] == 1 for row in saved), saved)


if __name__ == "__main__":
    unittest.main()
