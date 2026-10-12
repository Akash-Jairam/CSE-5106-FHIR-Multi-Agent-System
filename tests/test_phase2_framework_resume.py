"""Framework dependency upgrades invalidate previously saved run identity."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_agent as runner


class DependencyFingerprintTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "run_agent.py").write_text("# runner fixture\n", encoding="utf-8")
        (self.root / "fhir_client.py").write_text("# client fixture\n", encoding="utf-8")
        self.requirements = self.root / "requirements.txt"
        self.requirements.write_text("langgraph==1.2.14\nRequests==2.32.5\n", encoding="utf-8")
        self.versions = {"langgraph": "1.2.14", "Requests": "2.32.5", "langchain-core": "1.6.9",
                         "langgraph-checkpoint": "4.2.0", "langgraph-prebuilt": "1.1.0",
                         "langgraph-sdk": "0.4.6", "langsmith": "0.14.7"}

    def fingerprint(self):
        with patch.object(runner, "__file__", str(self.root / "run_agent.py")), \
                patch.object(runner.metadata, "version", side_effect=lambda name: self.versions[name]):
            return runner.code_fingerprint()

    def test_pin_changes_invalidate_resume_identity(self):
        previous = self.fingerprint()
        self.requirements.write_text("langgraph==1.2.13\nRequests==2.32.5\n", encoding="utf-8")
        self.assertNotEqual(previous, self.fingerprint())

    def test_installed_graph_and_transitive_upgrades_invalidate_identity(self):
        previous = self.fingerprint()
        for name in ("langgraph", "langchain-core", "langsmith"):
            original = self.versions[name]
            self.versions[name] = "different-version"
            self.assertNotEqual(previous, self.fingerprint(), name)
            self.versions[name] = original
        self.assertEqual(previous, self.fingerprint())

    def test_missing_dependency_is_explicit_in_the_fingerprint(self):
        with patch.object(runner.metadata, "version", side_effect=runner.metadata.PackageNotFoundError):
            versions = runner.runtime_dependency_versions(self.requirements)
        self.assertIn("langgraph", versions)
        self.assertIsNone(versions["langgraph"])

    def test_only_declared_public_dependency_metadata_is_read(self):
        with patch.object(runner.metadata, "version", return_value="fixture") as version:
            versions = runner.runtime_dependency_versions(self.requirements)
        self.assertEqual(set(versions), set(self.versions))
        self.assertEqual({call.args[0] for call in version.call_args_list}, set(self.versions))


if __name__ == "__main__":
    unittest.main()
