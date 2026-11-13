"""The V10 router recovers the finisher templates from question text, exactly, on the whole benchmark."""
import collections
import csv
import sys
import unittest
from pathlib import Path

from agent.template_router import classify_template

ROOT = Path(__file__).resolve().parent.parent
TARGETS = {"T001", "T051", "T052", "T054", "T055", "T064", "T073"}


class TemplateRouterTest(unittest.TestCase):
    def test_exact_on_every_benchmark_question(self):
        csv.field_size_limit(sys.maxsize)
        with open(ROOT / "manifests/question_families.csv", newline="") as stream:
            templates = {r["question_id"]: r["template_id"] for r in csv.DictReader(stream)}
        with open(ROOT / "final_dataset/questions_answers_sql_fhir.csv", newline="") as stream:
            questions = {r["question_id"]: r["question"] for r in csv.DictReader(stream)}
        confusion = collections.Counter()
        for qid, template in templates.items():
            expected = template if template in TARGETS else None
            got = classify_template(questions[qid])
            if got != expected:
                confusion[(template, got)] += 1
        self.assertEqual(len(templates), 2931)
        self.assertEqual(dict(confusion), {})

    def test_near_miss_wordings(self):
        cases = {
            "When did patient 1 receive a urine microbiology test the last time?": None,        # T059
            "What was the name of the microbiology test that patient 1 first received?": None,  # T053
            "What was the dose of heparin that patient 1 was first prescribed?": None,          # T050
            "When was the last time that patient 1 was prescribed heparin?": None,              # T074
            "First time, what drug was prescribed for patient 1 in 07/this year?": "T051",
            "What culture was patient 1 last tested since 02/2129?": "T054",
            "Has there been any organism found in the first urine test of patient 1?": None,    # T022
        }
        for question, expected in cases.items():
            self.assertEqual(classify_template(question), expected, question)


if __name__ == "__main__":
    unittest.main()
