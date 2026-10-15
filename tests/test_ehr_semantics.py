"""The shared guide is gold-free and reaches every competing system (B3)."""
import re
import unittest

from agent.ehr_semantics import CONVENTION_GUIDANCE, SEMANTIC_GUIDANCE


class ConventionGuidanceTest(unittest.TestCase):

    def test_conventions_are_part_of_the_shared_guide(self):
        self.assertTrue(SEMANTIC_GUIDANCE.endswith(CONVENTION_GUIDANCE))
        for phrase in ("Specimen", "hasMember", "shares the selected timestamp", "distinct released name",
                       "priority coding", "plain list"):
            self.assertIn(phrase, CONVENTION_GUIDANCE)

    def test_guide_contains_no_question_or_patient_identifiers(self):
        # 24-hex question ids, MIMIC subject ids, deidentified years
        self.assertIsNone(re.search(r"\b[0-9a-f]{24}\b|\b100\d{5}\b|\b2[12]\d\d-\d\d-\d\d\b", SEMANTIC_GUIDANCE))

    def test_every_system_prompt_carries_the_conventions(self):
        from agent.direct_llm_agent import DirectLLMAgent
        from agent.single_agent import SINGLE_SYSTEM_PROMPT
        from agent.query_contract_agent import AUDIT_PROMPT, CONTRACT_PROMPT

        self.assertIn(CONVENTION_GUIDANCE, DirectLLMAgent("m").system_msg["content"])
        self.assertIn(CONVENTION_GUIDANCE, SINGLE_SYSTEM_PROMPT)
        self.assertIn(CONVENTION_GUIDANCE, CONTRACT_PROMPT)
        self.assertIn(CONVENTION_GUIDANCE, AUDIT_PROMPT)


if __name__ == "__main__":
    unittest.main()
