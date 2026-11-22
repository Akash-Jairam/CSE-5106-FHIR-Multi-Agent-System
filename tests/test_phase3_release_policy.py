"""Offline release rules for multi_agent's verification gate (scripts/release_policy.py)."""
import json
import unittest

from scripts.release_policy import released_answer


def row(answer=None, draft=None, final=None, check_issues=None, final_check_issues=None, as_text=False):
    revision = {"draft": draft, "final": final, "check_issues": check_issues, "final_check_issues": final_check_issues}
    return {"agent_answer": answer, "revision": json.dumps(revision) if as_text else revision}


def issue(kind):
    return {"type": kind, "record": None, "field": None, "responsible_role": None, "action": ""}


class ReleaseRuleTests(unittest.TestCase):
    def test_strict_keeps_the_system_answer(self):
        withheld = row(None, draft={"answer": 3}, check_issues=[issue("recompute_mismatch")])
        self.assertIsNone(released_answer(withheld, "strict"))
        self.assertEqual(released_answer(row(4, final={"answer": 4}, final_check_issues=[]), "strict"), 4)

    def test_field_check_ignores_model_judgment_issues(self):
        judged = row(None, draft={"answer": 3},
                     check_issues=[issue("recompute_mismatch"), issue("missing_evidence"), issue("independent_check")])
        self.assertEqual(released_answer(judged, "field_check"), 3)

    def test_field_check_withholds_on_a_deterministic_issue(self):
        mismatched = row(None, draft={"answer": 3}, check_issues=[issue("value_mismatch"), issue("wrong_filter")])
        self.assertIsNone(released_answer(mismatched, "field_check"))
        self.assertEqual(released_answer(mismatched, "always"), 3)

    def test_the_final_answer_and_its_issues_take_precedence_over_the_draft(self):
        revised = row(None, draft={"answer": 3}, check_issues=[issue("value_mismatch")],
                      final={"answer": 5}, final_check_issues=[issue("wrong_calculation")], as_text=True)
        self.assertEqual(released_answer(revised, "field_check"), 5)
        self.assertEqual(released_answer(revised, "always"), 5)

    def test_nothing_to_release_without_a_draft(self):
        self.assertIsNone(released_answer(row(None), "always"))
        self.assertIsNone(released_answer({"agent_answer": None, "revision": "not json"}, "always"))


if __name__ == "__main__":
    unittest.main()
