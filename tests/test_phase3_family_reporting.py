"""Analysis labels remain bound to the frozen source-family manifest."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from scripts import run_experiments as experiments


class FamilyReportingTests(unittest.TestCase):
    def test_changed_frozen_families_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            families = root / "families.csv"
            families.write_text("question_id,family,template\na,lookup_counting,count\n")
            frozen_hash = hashlib.sha256(families.read_bytes()).hexdigest()
            split = root / "split.json"
            split.write_text(json.dumps({"family_manifest_sha256": frozen_hash}))
            families.write_text("question_id,family,template\na,temporal_selection,count\n")
            args = SimpleNamespace(input="unused", families=str(families), split_manifest=str(split))
            with patch.object(experiments.run_agent, "read_input_data",
                              return_value=pd.DataFrame([{"question_id": "a"}])):
                with self.assertRaisesRegex(ValueError, "frozen with the split"):
                    experiments.load_references(args, {"a"})

    def test_duplicate_or_missing_family_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split.json"
            split.write_text("{}")
            families = root / "families.csv"
            args = SimpleNamespace(input="unused", families=str(families), split_manifest=str(split))
            for records in ("a,lookup_counting,count\na,lookup_counting,count\n",
                            "b,lookup_counting,count\n"):
                families.write_text("question_id,family,template\n" + records)
                with patch.object(experiments.run_agent, "read_input_data",
                                  return_value=pd.DataFrame([{"question_id": "a"}])):
                    with self.subTest(records=records), self.assertRaises(ValueError):
                        experiments.load_references(args, {"a"})


if __name__ == "__main__":
    unittest.main()
