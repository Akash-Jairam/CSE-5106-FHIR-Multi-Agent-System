"""P3-19: run orchestration for 900 ordinary and 200 recovery runs.

Covers the multi-agent strategy, the omitted-packet condition, the runner's
perturbation options and resume identity, the experiment plan, status, and
resume-by-rerun, and the tool cache under parallel workers. No model or FHIR
store is called.
"""
import contextlib
import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

import run_agent as runner
from agent.direct_llm_agent import DirectLLMAgent
from agent.multi_agent import MultiAgentSystem
from agent.single_agent import SingleAgent
from scripts import run_experiments as experiments
from tools import cache
from tools.starting_packet import PACKET_MAX_TOKENS
from utils import core_utils, curate_input_dataset, parse_question_context

PATIENT = "11111111-1111-1111-1111-111111111111"
NOW = "2153-12-31 23:59:00"
QUESTION = f"Question: What was the last weight?\nContext:\nPatient FHIR ID is {PATIENT}.\nAssume current time is {NOW}."
SPEC = {"question_id": "q2", "removed": {"resource_type": "Observation", "id": "o1"}}
PACKET = {"text": "Starting packet", "resources": {}, "resource_ids": {}, "n_tokens": 2, "truncated": False}


def inputs(ids=("q1", "q2", "q3")):
    return pd.DataFrame([{"question_id": qid, "question": "Question " + qid, "true_answer": "7",
                          "assumption": None, "patient_fhir_id": "patient"} for qid in ids])


class PerturbedPacketTests(unittest.TestCase):
    """Each system that loads a starting packet uses the frozen omission when one is set."""

    def run_system(self, cls, module, perturbation):
        with patch("agent.base_agent.setup_api_keys"), \
                patch(f"agent.{module}.get_starting_packet", return_value=PACKET) as intact, \
                patch(f"agent.{module}.get_perturbed_starting_packet", return_value=PACKET) as omitted:
            agent = cls("test-model")
            agent.set_perturbation(perturbation)
            agent._llm_call = Mock(return_value=(None, "provider down", None))
            agent.run(QUESTION)
        return intact, omitted

    def test_every_packet_loader_honours_the_perturbation(self):
        for cls, module in ((SingleAgent, "single_agent"), (DirectLLMAgent, "direct_llm_agent"),
                            (MultiAgentSystem, "retrieval_agent")):
            with self.subTest(system=cls.__name__):
                intact, omitted = self.run_system(cls, module, SPEC)
                intact.assert_not_called()
                omitted.assert_called_once_with("What was the last weight?", PATIENT, SPEC, current_time=NOW)
                intact, omitted = self.run_system(cls, module, None)
                intact.assert_called_once_with("What was the last weight?", PATIENT, current_time=NOW)
                omitted.assert_not_called()

    def test_perturbation_must_be_a_frozen_specification(self):
        with patch("agent.base_agent.setup_api_keys"):
            with self.assertRaises(ValueError):
                SingleAgent("test-model").set_perturbation([("Observation", "o1")])

    def test_worker_passes_the_perturbation_to_the_agent(self):
        agent = Mock()
        with patch.object(core_utils, "create_agent", return_value=agent), \
                patch.object(core_utils, "run_agent_safe", return_value={}) as run:
            core_utils.run_agent_with_input((QUESTION, "multi_agent", "m", False, None, True, {"max_tool_calls": 3}, SPEC))
        agent.set_budget.assert_called_once_with(max_tool_calls=3)
        agent.set_perturbation.assert_called_once_with(SPEC)
        run.assert_called_once_with(agent, QUESTION)

    def test_multi_agent_strategy_builds_the_three_roles(self):
        with patch("agent.base_agent.setup_api_keys"):
            system = core_utils.create_agent("multi_agent", "test-model")
            context = system.question_context(QUESTION)
            roles = system.build_roles(context, {})
        self.assertIsInstance(system, MultiAgentSystem)
        self.assertEqual(set(roles), {"retrieval", "analysis", "verification"})


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manifest = os.path.join(self.temp.name, "perturbations.json")
        with open(self.manifest, "w") as stream:
            json.dump({"version": 2}, stream)

    def args(self, *extra):
        return ["--agent_strategy", "multi_agent", "--model", "test-model", "--num_processes", "1",
                "--output", os.path.join(self.temp.name, "out.json"), *extra]

    def test_perturbation_options_are_validated(self):
        for extra in (["--perturbations", self.manifest],
                      ["--condition", "omitted"],
                      ["--perturbations", self.manifest, "--condition", "omitted"],
                      ["--perturbations", self.manifest, "--condition", "omitted",
                       "--split_manifest", "manifests/split.json", "--split", "pilot"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(self.args(*extra))

    def test_perturbed_questions_keep_split_order_before_the_limit(self):
        with patch.object(runner, "read_input_data", return_value=inputs()):
            selected = runner.load_inputs("unused.csv", limit=1, question_ids=["q3", "q2"])
            self.assertEqual(selected["question_id"].tolist(), ["q2"])
            with self.assertRaisesRegex(ValueError, "not in the selected inputs"):
                runner.load_inputs("unused.csv", question_ids=["q9"])

    def test_condition_and_manifest_are_part_of_the_resume_identity(self):
        base = {"agent_strategy": "multi_agent", "model": "m", "base_url": None, "max_total_tokens": None,
                "max_tool_calls": None, "add_patient_fhir_id": True, "enable_cache": True}
        other = os.path.join(self.temp.name, "other.json")
        with open(other, "w") as stream:
            json.dump({"version": 3}, stream)
        identities = [runner.run_identity(SimpleNamespace(**base, **extra), inputs()) for extra in (
            {}, {"perturbations": self.manifest, "condition": "intact"},
            {"perturbations": self.manifest, "condition": "omitted"},
            {"perturbations": other, "condition": "omitted"})]
        self.assertEqual(len({json.dumps(i, sort_keys=True) for i in identities}), 4)

    def run_condition(self, condition):
        specs = {"q2": SPEC, "q3": {**SPEC, "question_id": "q3"}}
        outputs = []

        def worker(job):
            outputs.append(job)
            return {"agent_answer": None, "latency": 1.0, "error": None}

        with patch("tools.perturbation.load_perturbations", return_value=specs) as load, \
                patch.object(runner, "read_input_data", return_value=inputs()), \
                patch.object(runner, "select_split", side_effect=lambda df, *_: df), \
                patch.object(runner, "validate_config", return_value=True), \
                patch.object(runner, "check_tool_credentials"), \
                patch.object(runner, "run_agent_with_input", side_effect=worker), \
                contextlib.redirect_stdout(io.StringIO()):
            split = os.path.join(self.temp.name, "split.json")
            with open(split, "w") as stream:
                json.dump({}, stream)
            code = runner.main(self.args("--perturbations", self.manifest, "--condition", condition, "--quiet",
                                         "--split_manifest", split, "--split", "held_out"))
        self.assertEqual(code, 0)
        load.assert_called_once_with(self.manifest)
        return outputs

    def test_omitted_runs_get_their_specification_and_intact_runs_none(self):
        omitted = self.run_condition("omitted")
        self.assertEqual([job[0].split("\n")[0] for job in omitted], ["Question: Question q2", "Question: Question q3"])
        self.assertEqual([job[-1]["question_id"] for job in omitted], ["q2", "q3"])
        os.remove(os.path.join(self.temp.name, "out.json"))
        self.assertEqual([job[-1] for job in self.run_condition("intact")], [None, None])


class FrozenContextTests(unittest.TestCase):
    def test_every_frozen_perturbation_matches_the_context_agents_receive(self):
        from agent.controller import FixedController
        from tools.perturbation import load_perturbations, validate_context

        specs = load_perturbations()
        frame = runner.load_inputs(experiments.DEFAULT_INPUT, None, experiments.DEFAULT_SPLIT, "held_out",
                                   question_ids=list(specs))
        self.assertEqual(len(frame), 50)
        for qid, text in zip(frame["question_id"], curate_input_dataset(frame, True)):
            question, patient, now = parse_question_context(text)
            validate_context(question, patient, specs[qid], PACKET_MAX_TOKENS, now)
            context = FixedController.question_context(text)
            validate_context(context["question"], context["patient_fhir_id"], specs[qid], PACKET_MAX_TOKENS,
                             context["current_time"])


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def cli(self, *extra):
        return ["--model", "test-model", "--output_dir", self.temp.name, "--max_tool_calls", "5", *extra]

    def test_the_full_plan_is_900_ordinary_and_200_recovery_runs(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(experiments.main(["plan", *self.cli()]), 0)
        self.assertIn("ordinary: 0/900 runs complete", out.getvalue())
        self.assertIn("recovery: 0/200 runs complete", out.getvalue())
        self.assertEqual([arm.name for arm in experiments.ARMS], [
            "ordinary_direct_llm", "ordinary_single_agent", "ordinary_multi_agent",
            "recovery_single_agent_intact", "recovery_single_agent_omitted",
            "recovery_multi_agent_intact", "recovery_multi_agent_omitted"])

    def test_every_arm_shares_model_budget_and_cache(self):
        args = experiments.build_parser().parse_args(["run", *self.cli("--base_url", "http://llm")])
        for arm in experiments.ARMS:
            settings = runner.build_parser().parse_args(experiments.runner_argv(arm, args))
            self.assertEqual((settings.agent_strategy, settings.model, settings.base_url, settings.max_tool_calls,
                              settings.enable_cache, settings.split, settings.condition),
                             (arm.strategy, "test-model", "http://llm", 5, True, "held_out", arm.condition))
            self.assertEqual(settings.perturbations is not None, arm.group == "recovery")
            self.assertEqual(settings.output, os.path.join(self.temp.name, arm.name + ".json"))

    def test_arm_selection(self):
        names = lambda *selected: [arm.name for arm in experiments.select_arms(selected, "held_out")]
        self.assertEqual(names("recovery_multi_agent_omitted", "ordinary_direct_llm"),
                         ["ordinary_direct_llm", "recovery_multi_agent_omitted"])
        self.assertEqual(len(names("recovery")), 4)
        with self.assertRaisesRegex(ValueError, "Unknown arms"):
            experiments.select_arms(["everything"], "held_out")
        with self.assertRaisesRegex(ValueError, "held-out perturbations"):
            experiments.select_arms(["recovery"], "pilot")

    def status(self, *extra):
        args = experiments.build_parser().parse_args(["status", *self.cli("--arms", "ordinary_direct_llm", *extra)])
        return experiments.arm_status(experiments.ARMS[0], args, {})

    def test_status_counts_done_failed_and_stale_files(self):
        self.assertEqual(self.status("--limit", "3")["state"], "not started")
        args = experiments.build_parser().parse_args(["status", *self.cli("--limit", "3")])
        settings = runner.build_parser().parse_args(experiments.runner_argv(experiments.ARMS[0], args))
        frame = experiments.arm_inputs(experiments.ARMS[0], settings, {})
        results = runner.prepare_results(frame, settings.output, runner.run_identity(settings, frame))
        results.loc[0, ["agent_answer", "latency"]] = [7, 1.0]
        results.loc[1, ["latency", "error"]] = [1.0, "failed"]
        with contextlib.redirect_stdout(io.StringIO()):
            runner.save_results_json(results, settings.output)
        runner._write_json(runner.run_identity(settings, frame), settings.output + ".run.json")
        status = self.status("--limit", "3")
        self.assertEqual((status["completed"], status["failed"], status["questions"], status["state"]),
                         (1, 1, 3, "in progress"))
        self.assertIn("stale", self.status("--limit", "3", "--max_total_tokens", "9")["state"])

    def test_run_continues_after_a_failed_arm_and_reports_incomplete(self):
        calls = []

        def fake_main(argv):
            calls.append(argv[argv.index("--agent_strategy") + 1])
            if len(calls) == 1:
                raise ValueError("stale results")
            return 0

        with patch.object(experiments.run_agent, "main", side_effect=fake_main), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            code = experiments.main(["run", *self.cli("--arms", "ordinary", "--split", "pilot")])
        self.assertEqual(code, 1)
        self.assertEqual(calls, list(experiments.SYSTEMS))
        self.assertIn("FAILED ordinary_direct_llm: ValueError: stale results", out.getvalue())
        self.assertIn("Rerun the same command", out.getvalue())


class ResultAccountingTests(unittest.TestCase):
    def test_saved_rows_are_validated_before_resume_or_pairing(self):
        frame = inputs(("q1", "q2"))
        valid = {"question_id": "q1", "patient_fhir_id": "patient", "latency": 1.0}
        invalid = ([valid, valid], [{**valid, "question_id": None}],
                   [{**valid, "question_id": "foreign"}],
                   [{**valid, "patient_fhir_id": "another-patient"}], {"q1": valid}, [None])
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "results.json")
            for records in invalid:
                runner._write_json(records, output)
                with self.subTest(records=records), self.assertRaises(ValueError):
                    runner.load_saved_results(output, frame)
            runner._write_json([valid], output)
            self.assertEqual(runner.load_saved_results(output, frame), {"q1": valid})

    def test_all_attempted_errors_are_analyzable_but_retryable(self):
        frame = inputs(("q1", "q2"))
        with tempfile.TemporaryDirectory() as directory:
            args = experiments.build_parser().parse_args([
                "status", "--model", "test", "--output_dir", directory, "--arms", "ordinary_direct_llm"])
            arm = experiments.ARMS[0]
            settings = runner.build_parser().parse_args(experiments.runner_argv(arm, args))
            records = [{**row, "latency": 1.0, "agent_answer": None,
                        "error": "provider down" if index else None}
                       for index, row in enumerate(frame.to_dict("records"))]
            runner._write_json(records, settings.output)
            runner._write_json(runner.run_identity(settings, frame), settings.output + ".run.json")
            with patch.object(experiments, "arm_inputs", return_value=frame):
                status = experiments.arm_status(arm, args, {})
            self.assertEqual((status["completed"], status["failed"], status["attempted"]), (1, 1, 2))
            self.assertTrue(status["analyzable"])
            self.assertNotEqual(status["state"], "complete")

    def test_full_run_requires_explicit_shared_ceilings_before_execution(self):
        with patch.object(runner, "main") as execute, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                experiments.main(["run", "--model", "test", "--arms", "ordinary"])
        execute.assert_not_called()


class CacheTests(unittest.TestCase):
    def test_cache_entries_are_written_atomically(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(cache, "CACHE_DIR", directory), \
                contextlib.redirect_stdout(io.StringIO()):
            cache.save_to_cache("tool", (), {"a": 1}, {"records": [1]})
            self.assertEqual(cache.load_from_cache("tool", (), {"a": 1}), {"records": [1]})
            self.assertEqual([name for name in os.listdir(directory) if not name.endswith(".cache")], [])


if __name__ == "__main__":
    unittest.main()
