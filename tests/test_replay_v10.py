"""Offline V10 replay and mechanism table on fixture rows (no answers are printed)."""
import unittest

from scripts.replay_v10 import mechanism, predict, signature

Q51 = "What was the name of the drug that patient 1 was first prescribed?"
Q54 = "What was the name of the specimen that patient 1 was first tested?"
REFERENCES = {
    "q0": {"question": Q51, "true_answer": "[['Heparin']\n ['Insulin']]", "template": ""},
    "q1": {"question": Q51, "true_answer": "[['Aspirin']]", "template": ""},
    "q2": {"question": Q54, "true_answer": "[['URINE']]", "template": ""},
    "q3": {"question": Q51, "true_answer": "[['Senna']]", "template": ""},
}
TEMPLATES = {"q0": "T051", "q1": "T051", "q2": "T054", "q3": "T051"}


def single(answer, evidence=(), tokens=40_000, tools=6):
    return {"agent_answer": answer, "error": None, "evidence": list(evidence), "tokens": {"total": tokens},
            "budget": {"tool_calls_run": tools}}


def team(answer, ids=(), tokens=40_000, tools=6):
    return {"agent_answer": answer, "error": None,
            "revision": {"primary": {"answer": answer, "evidence": [], "contributing_ids": list(ids)},
                         "phases": [{"phase": "primary", "tokens": tokens, "tool_calls": tools}]}}


RX = [{"resource_id": "MedicationRequest/rx1", "field_path": "id", "value": "rx1"}]


class ReplayV10Tests(unittest.TestCase):
    def test_signatures(self):
        self.assertEqual(signature("T051", "Heparin", {"MedicationRequest"}, REFERENCES["q0"]), "bare_tie_member")
        self.assertIsNone(signature("T051", "Warfarin", {"MedicationRequest"}, REFERENCES["q0"]))
        self.assertIsNone(signature("T051", "Aspirin", {"MedicationRequest"}, REFERENCES["q1"]))  # no tie
        self.assertEqual(signature("T054", "URINE CULTURE", {"Observation"}, REFERENCES["q2"]), "observation_only")
        self.assertIsNone(signature("T054", "URINE", {"Observation", "Specimen"}, REFERENCES["q2"]))

    def test_predict_counts_votes_reachability_and_never_prints_answers(self):
        singles = {"q0": single("Heparin", RX), "q1": single("Aspirin", RX), "q2": single("URINE CULTURE", [
            {"resource_id": "Observation/o1", "field_path": "id", "value": "o1"}]),
                   "q3": single("Senna", RX)}
        teams = {"q0": team("Heparin", ["MedicationRequest/rx1"]), "q1": team("Insulin", ["MedicationRequest/rx2"]),
                 "q2": team("URINE CULTURE", ["Observation/o1"]),
                 "q3": team("Warfarin", ["MedicationRequest/rx3"], tokens=150_000)}
        report = predict(singles, teams, REFERENCES, TEMPLATES, 200_000, 30)
        self.assertEqual(report["scores"], {"fresh_single": 2, "team_primary_draw1": 0, "oracle_two_draws": 2})
        pairs = report["two_draws"]["pairs"]
        self.assertEqual(pairs["agree_wrong"], 2)
        self.assertEqual(pairs["disagree_draw2_right"], 2)
        self.assertEqual(report["two_draws"]["availability"],
                         {"draw2_available": 4, "draw3_available": 1, "draw3_unavailable": 1})
        self.assertEqual(report["two_draws"]["expected_vote_change_over_draw1"], 0.5)
        reach = report["finisher_reachability"]["fresh_single"]["total"]
        self.assertEqual((reach["wrong"], reach["reachable"]), (2, 2))
        self.assertEqual(report["router_check"], {"agree": 4})
        for name in ("Heparin", "Insulin", "Aspirin", "URINE", "Senna", "Warfarin"):
            self.assertNotIn(name, repr(report))

    def test_mechanism_counts_finisher_replacements_and_the_gate(self):
        rows = {"q0": {"agent_answer": ["Heparin", "Insulin"], "error": None, "tokens": {"total": 90_000},
                       "revision": {"protocol": "finisher-team-v10", "release_policy": "finisher_replaced",
                                    "vote": "agree2", "draft": {"answer": "Heparin"},
                                    "candidate": {"answer": "Heparin"},
                                    "finisher": {"template": "T051", "finisher": "replaced", "reason": None},
                                    "phases": [{"phase": "draw1", "tokens": 40_000}, {"phase": "draw2", "tokens": 50_000},
                                               {"phase": "finisher", "tokens": 0}]}},
                "q1": {"agent_answer": "Insulin", "error": None, "tokens": {"total": 80_000},
                       "revision": {"protocol": "finisher-team-v10", "release_policy": "finisher_replaced",
                                    "vote": "split", "draft": {"answer": "Aspirin"}, "candidate": {"answer": "Aspirin"},
                                    "finisher": {"template": "T051", "finisher": "replaced", "reason": None}}},
                "q2": {"agent_answer": "URINE", "error": None, "tokens": {"total": 60_000},
                       "revision": {"protocol": "finisher-team-v10", "release_policy": "vote_candidate",
                                    "vote": "majority3", "draft": {"answer": "BLOOD"},
                                    "finisher": {"template": "T054", "finisher": "confirmed_primary",
                                                 "reason": "result_equals_primary"}}},
                "q3": {"agent_answer": None, "error": "provider 520", "tokens": {"total": 1_000}, "revision": None}}
        report = mechanism(rows, REFERENCES, TEMPLATES)
        self.assertEqual(report["finisher_replacements"]["total"], {"wrong_to_right": 1, "right_to_wrong": 1})
        self.assertTrue(report["gate_finisher_harm_ok"])
        self.assertEqual(report["policies"]["terminal_error"], 1)
        self.assertEqual(report["finisher_reasons"]["T054"], {"result_equals_primary": 1})
        self.assertEqual((report["draw1_correct"], report["released_correct"]), (1, 2))
        self.assertEqual(report["unknown_policies"], [])
        self.assertEqual(report["unknown_votes"], ["None"])
        self.assertEqual(report["tokens"]["finisher"]["n"], 1)
