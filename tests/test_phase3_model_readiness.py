"""Selected-model compatibility cannot depend on registry freshness or expose secrets."""
import tempfile
import traceback
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import core_utils, model_config


class ModelReadinessTests(unittest.TestCase):
    def test_known_luna_remains_reasoning_when_registry_is_missing(self):
        with patch.object(core_utils.litellm, "supports_reasoning", return_value=False):
            self.assertTrue(core_utils.is_reasoning_llm("gpt-6-luna"))
            self.assertTrue(core_utils.is_reasoning_llm("openai/gpt-6-luna"))
            self.assertTrue(core_utils.is_reasoning_llm("o4-mini"))
            self.assertFalse(core_utils.is_reasoning_llm("gpt-4.1-nano"))

    def test_malformed_config_error_does_not_echo_secret_source_lines(self):
        marker = "SYNTHETIC_SECRET_MUST_NOT_APPEAR"
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yml"
            config.write_text("OPENAI_API_KEY: [" + marker + "\n")
            with patch.object(model_config, "CONFIG_PATH", config), \
                    patch.dict("os.environ", {"MODEL": ""}):
                try:
                    model_config.get_default_model()
                except ValueError as error:
                    diagnostic = "".join(traceback.format_exception(error))
                    self.assertIn("Could not read", diagnostic)
                    self.assertNotIn(marker, diagnostic)
                    self.assertNotIn("OPENAI_API_KEY", diagnostic)
                else:
                    self.fail("Malformed configuration was silently accepted")


if __name__ == "__main__":
    unittest.main()
