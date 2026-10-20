"""Offline V9 replay and mechanism table on fixture rows (no answers are printed)."""
import unittest

from scripts.replay_v9_controller import mechanism, recompute_candidate, replay

REFERENCES = {f"q{i}": {"question": "What was the first drug?", "true_answer": "['Heparin']", "template": ""}
              for i in range(6)}


def single(answer, tokens=50_000, tools=8, error=None):
    return {"agent_answer": answer, "error": error, "tokens": {"total": tokens},
            "budget": {"tool_calls_run": tools, "stopped": None}}


def v8_team(answer, solver_tokens=40_000, solver_tools=6, issues=()):
    return {"agent_answer": answer, "error": None,
            "revision": {"primary": {"answer": answer, "evidence": [], "contributing_ids": ["MedicationRequest/rx1"]},
                         "draft": {"answer": answer}, "final_check_issues": list(issues),
                         "primary_revision": {"check_issues": list(issues)}, "unresolved_scope": {},
                         "phases": [{"phase": "solver", "tokens": solver_tokens, "tool_calls": solver_tools}]}}


class ReplayTests(unittest.TestCase):
    def test_policies_scores_and_availability(self):
        singles = {"q0": single("Heparin"), "q1": single(None), "q2": single("Insulin"),
                   "q3": single(None, tokens=190_000), "q4": single("Heparin"), "q5": single(None)}
        teams = {"q0": v8_team("heparin"), "q1": v8_team("Heparin"), "q2": v8_team("Heparin"),
                 "q3": v8_team("Heparin"), "q4": v8_team("Heparin", solver_tokens=170_000),
                 "q5": v8_team("Heparin", issues=["evidence[0] value_mismatch"])}
        report = replay(singles, teams, REFERENCES, 200_000, 30)
        self.assertEqual(report["policies"]["agree_primary"], 1)
        self.assertEqual(report["policies"]["null_rule_recompute"], 1)
        self.assertEqual(report["policies"]["null_rule_recompute_right"], 1)
        self.assertEqual(report["policies"]["adjudicate"], 1)
        self.assertEqual(report["policies"]["null_rule_primary"], 1)  # q5: recompute not source-valid
        self.assertEqual(report["policies"]["recompute_skipped_budget"], 2)  # q3 residual, q4 truncated
        self.assertEqual(report["recompute_availability"],
                         {"available": 4, "skipped_residual": 1, "truncated_estimate": 1})
        self.assertEqual(report["adjudication_cases"], {"primary_wrong/recompute_right": 1})
        scores = report["scores"]
        self.assertEqual(scores["single"], 2)
        self.assertEqual(scores["predicted_default_primary"], 3)
        self.assertEqual(scores["bound_adjudication_oracle"], 4)
        self.assertEqual(scores["oracle_pick"], 6)
        self.assertNotIn("Heparin", repr(report))

    def test_draft_field_uses_the_draft_check(self):
        row = v8_team("Heparin")
        row["revision"]["draft"] = {"answer": "Insulin"}
        row["revision"]["primary_revision"]["check_issues"] = ["evidence[0] value_mismatch"]
        self.assertEqual(recompute_candidate(row, "draft"), {"answer": "Insulin", "source_valid": False})
        self.assertEqual(recompute_candidate(row), {"answer": "Heparin", "source_valid": True})

    def test_mismatched_cohorts_are_refused(self):
        with self.assertRaises(ValueError):
            replay({"q0": single("x")}, {}, REFERENCES, 200_000, 30)

    def test_mechanism_counts_transitions_per_policy(self):
        rows = {"q0": {"agent_answer": "Heparin", "error": None, "tokens": {"total": 90_000},
                       "revision": {"protocol": "dual-solver-v9", "release_policy": "null_rule_recompute",
                                    "draft": {"answer": None}, "primary_error": None,
                                    "phases": [{"phase": "primary", "tokens": 60_000},
                                               {"phase": "recompute", "tokens": 30_000}]}},
                "q1": {"agent_answer": "Insulin", "error": None, "tokens": {"total": 120_000},
                       "revision": {"protocol": "dual-solver-v9", "release_policy": "adjudicated_recompute",
                                    "draft": {"answer": "Heparin"},
                                    "adjudication": {"check_status": "valid", "error": None},
                                    "phases": [{"phase": "primary", "tokens": 60_000},
                                               {"phase": "recompute", "tokens": 50_000},
                                               {"phase": "adjudicate", "tokens": 10_000}]}},
                "q2": {"agent_answer": None, "error": "provider 520", "tokens": {"total": 1_000},
                       "revision": None}}
        report = mechanism(rows, REFERENCES)
        self.assertEqual(report["policies"], {"adjudicated_recompute": 1, "null_rule_recompute": 1,
                                              "terminal_error": 1})
        self.assertEqual(report["replacements"]["null_rule_recompute"], {"wrong_to_right": 1})
        self.assertEqual(report["replacements"]["adjudicated_recompute"], {"right_to_wrong": 1})
        self.assertEqual(report["adjudication"]["check_valid"], 1)
        self.assertEqual(report["tokens"]["recompute"]["n"], 2)
        self.assertEqual(report["unknown_policies"], [])


if __name__ == "__main__":
    unittest.main()
