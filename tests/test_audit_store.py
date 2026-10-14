"""Optional compressed audits are lossless and never alter inference/scoring fields."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_store import externalize_result_audit, restore_result_audit


class AuditStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = {"agent_answer": 7, "evidence": [{"resource_id": "Observation/o1", "value": 7}],
                       "tokens": {"total": 10, "cost": None}, "budget": {"stopped": None}, "error": None,
                       "trace": [{"role": "tool", "content": "large Unicode 測 data " * 10000}],
                       "revision": {"draft": {"answer": 8}, "final": {"answer": 7}, "revised": True,
                                    "messages": [{"payload": {"records": list(range(10000))}}]}}

    def test_exact_round_trip_and_inline_scoring_contract(self):
        original = deepcopy(self.output)
        compact = externalize_result_audit(self.output, self.root)
        self.assertEqual(self.output, original)
        self.assertEqual(restore_result_audit(compact, self.root), original)
        for key in ("agent_answer", "evidence", "tokens", "budget", "error"):
            self.assertEqual(compact[key], original[key])
        for key in ("draft", "final", "revised"):
            self.assertEqual(compact["revision"][key], original["revision"][key])
        self.assertIsInstance(compact["trace"], list)
        self.assertLess(len(json.dumps(compact)), len(json.dumps(original)) // 100)

    def test_content_addressing_deduplicates_and_is_idempotent(self):
        first = externalize_result_audit(self.output, self.root)
        second = externalize_result_audit(self.output, self.root)
        self.assertEqual(first, second)
        self.assertEqual(externalize_result_audit(first, self.root), first)
        self.assertEqual(len(list((self.root / "audit").glob("*.json.gz"))), 1)

    def test_corruption_and_path_escape_are_rejected(self):
        compact = externalize_result_audit(self.output, self.root)
        reference = compact["trace"][0]["artifact_ref"]
        path = self.root / reference["path"]
        with gzip.open(path, "wb") as stream:
            stream.write(b'{}')
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            restore_result_audit(compact, self.root)
        reference["path"] = "../outside.json.gz"
        with self.assertRaisesRegex(ValueError, "reference"):
            restore_result_audit(compact, self.root)

    def test_empty_audit_does_not_create_artifacts(self):
        result = {"agent_answer": None, "trace": [], "revision": None}
        self.assertEqual(externalize_result_audit(result, self.root), result)
        self.assertEqual(restore_result_audit(result, self.root), result)
        self.assertFalse((self.root / "audit").exists())


if __name__ == "__main__":
    unittest.main()
