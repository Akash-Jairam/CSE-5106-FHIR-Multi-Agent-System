"""Guided baselines (scripts/run_guided_baselines.py): each guided prompt is its baseline's prompt plus the
analyst's shared guidance blocks, verbatim, and nothing else; the baselines' module constants are untouched."""
import unittest
from unittest.mock import patch

from agent import direct_llm_agent, retrieval_agent
from agent.direct_llm_agent import DirectLLMAgent
from agent.ehr_semantics import SEMANTIC_GUIDANCE
from agent.multi_turn_code_resource_agent import MultiTurnCodeResourceAgent
from agent.single_agent import SingleAgent
from scripts.run_guided_baselines import DirectLLMGuided, SingleAgentGuided

DIRECT_PROMPT = direct_llm_agent.SYSTEM_PROMPT


def build(cls):
    with patch("agent.base_agent.setup_api_keys"):
        return cls("test-model")


def prompt(agent):
    message = agent.system_msg[0] if isinstance(agent.system_msg, list) else agent.system_msg
    return message["content"]


def squeeze(text):
    return " ".join(text.split())


class GuidedPrompts(unittest.TestCase):
    cases = [(SingleAgentGuided, SingleAgent, (MultiTurnCodeResourceAgent.search_guidance, SEMANTIC_GUIDANCE)),
             (DirectLLMGuided, DirectLLMAgent, (SEMANTIC_GUIDANCE,))]

    def test_each_guidance_block_is_included_verbatim(self):
        for guided, _, blocks in self.cases:
            text = prompt(build(guided))
            for block in blocks:
                self.assertTrue(block.strip())
                self.assertIn(block, text, guided.__name__)

    def test_deleting_the_blocks_gives_back_the_unguided_prompt(self):
        for guided, baseline, blocks in self.cases:
            text = prompt(build(guided))
            for block in blocks:
                text = text.replace(block, "", 1)
            self.assertEqual(squeeze(text), squeeze(prompt(build(baseline))), guided.__name__)

    def test_direct_guided_has_no_search_guidance(self):
        self.assertNotIn(MultiTurnCodeResourceAgent.search_guidance.strip(), prompt(build(DirectLLMGuided)))

    def test_baselines_and_packet_reading_are_unchanged(self):
        build(SingleAgentGuided), build(DirectLLMGuided)
        self.assertEqual(direct_llm_agent.SYSTEM_PROMPT, DIRECT_PROMPT)
        self.assertIs(retrieval_agent.DIRECT_PROMPT, direct_llm_agent.SYSTEM_PROMPT)
        self.assertEqual(prompt(build(DirectLLMAgent)), DIRECT_PROMPT)
        self.assertEqual(SingleAgent.search_guidance, "")
        self.assertNotIn(SEMANTIC_GUIDANCE, prompt(build(SingleAgent)))


if __name__ == "__main__":
    unittest.main()
