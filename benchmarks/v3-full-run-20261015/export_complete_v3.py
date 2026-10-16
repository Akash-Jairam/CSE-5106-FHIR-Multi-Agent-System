"""Deliver the complete V3 arm while the matched benchmark remains unfinished.

This additive helper is outside the frozen runtime/source lists. It validates
the entire 2,931-question first-clean V3 cohort before importing the scorer or
analyzer. Only a NEW output directory is written. Frozen partial analysis runs
on a detached V3-only snapshot; controls, scheduling and live artifacts are
never changed. Its unchanged global partial output is retained alongside a
separate V3_COMPLETE_CONTROLS_PENDING delivery and receipt.
"""
import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.audit_store import ARTIFACT_PATTERN, restore_result_audit
from scripts.full_benchmark import iter_result_rows

QUESTIONS = 2931
ARM = "multi_agent"
STATUS = "V3_COMPLETE_CONTROLS_PENDING"
APPROVED_CLASSIFIER = {"version": "v3-infrastructure-v3",
                       "sha256": "370ffcb0738b384bafbe82a194dcaf2a5c1e9da1a025d2ea3abcfca8474dcccd"}
INPUT_COLUMNS = ["question_id", "question", "true_answer", "assumption", "patient_fhir_id"]
REQUIRED_SOURCES = ("run_agent.py", "scripts/v3_full_run.py", "scripts/v3_full_analysis.py",
                    "scripts/audit_store.py", "scripts/full_benchmark.py", "utils/answer_scorer.py",
                    "utils/evidence_scorer.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def input_identity_hash(path):
    # Same CSV selection/serialization as frozen read_input_data/run_identity;
    # no run_agent import, configuration loading, or answer evaluation occurs.
    import pandas as pd
    return hashlib.sha256(pd.read_csv(path)[INPUT_COLUMNS].to_json(
        orient="records", date_format="iso").encode()).hexdigest()


def attempted(row):
    def present(value):
        return value is not None and not (isinstance(value, float) and math.isnan(value))
    return ((present(row.get("error")) and row["error"] != "")
            or present(row.get("agent_answer")) or present(row.get("latency")))


def load_classifier(path):
    """Execute only classifier constants/helper definitions from pinned source."""
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    definitions = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
    selected, pending = set(), ["classify_attempt", "CLASSIFIER_VERSION"]
    while pending:
        name = pending.pop()
        if name in selected or name not in definitions:
            continue
        require(name != "ROOT", "Classifier unexpectedly depends on runner setup")
        selected.add(name)
        pending.extend(child.id for child in ast.walk(definitions[name])
                       if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
                       and child.id in definitions and child.id not in selected)
    allowed = {"CLASSIFIER_VERSION", "RUNTIME", "INTRINSIC", "PROVIDER_JWT_REJECTION",
               "diagnostic_errors", "intrinsic_tool_validation", "provider_jwt_paths", "classify_attempt"}
    require(selected <= allowed and "classify_attempt" in selected, "Unexpected classifier dependency")
    nodes = [node for node in tree.body if any(definitions[name] is node for name in selected)]
    require(not any(isinstance(node, ast.FunctionDef) and node.decorator_list for node in nodes),
            "Classifier decorators are not permitted")
    scope = {"re": re, "math": math, "hashlib": hashlib, "json": json,
             "restore_result_audit": restore_result_audit}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["classify_attempt"], scope["CLASSIFIER_VERSION"]


def v3_ledger(ledger):
    require(ledger.get("protocol") == "v3-first-runtime-clean-terminal"
            and ledger.get("selection") == "first_runtime_clean_terminal", "Acceptance protocol changed")
    return {"selections": (ledger.get("arms") or {}).get(ARM) or {},
            "counts": (ledger.get("counts") or {}).get(ARM) or {}}


class CompleteV3Gate:
    """Read-only proof of full cohort, first-clean selection and artifact ownership."""
    def __init__(self, run_dir):
        self.run = Path(run_dir).resolve()
        self.inventory = {}
        self._inode_cache = {}
        self.canonical = self.run / "full_multi_agent.json"
        self.identity_path = Path(str(self.canonical) + ".run.json")
        self.manifest_path = self.run / "run_manifest.json"
        self.ledger_path = self.run / "acceptance.json"

    def remember(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        digest = self._inode_cache.get(key)
        if digest is None:
            digest = self._inode_cache[key] = sha256(path)
        self.inventory[path] = {"sha256": digest, "bytes": stat.st_size}
        return digest

    def validate(self):
        self.manifest_bytes = self.manifest_path.read_bytes()
        self.manifest = json.loads(self.manifest_bytes)
        self.manifest_hash = self.remember(self.manifest_path)
        require(self.manifest.get("questions_per_arm") == QUESTIONS, "Run does not pin the 2,931-question cohort")
        self.ledger_bytes = self.ledger_path.read_bytes()
        ledger = json.loads(self.ledger_bytes)
        self.ledger_snapshot = v3_ledger(ledger)
        self.ledger_slice_hash = semantic_hash(self.ledger_snapshot)
        self.initial_ledger_hash = hashlib.sha256(self.ledger_bytes).hexdigest()
        self.initial_control_counts = {arm: len((ledger.get("arms") or {}).get(arm) or {})
                                       for arm in ("direct_llm", "single_agent")}
        source = Path(self.manifest["input"]["path"])
        require(self.remember(source) == self.manifest["input"]["sha256"], "Frozen dataset changed")
        self.source_rows, patients = {}, {}
        with source.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            require(set(INPUT_COLUMNS) <= set(reader.fieldnames or []), "Dataset input columns missing")
            for row in reader:
                qid, patient = row["question_id"], row["patient_fhir_id"]
                require(qid and patient and qid not in patients, "Dataset identities are missing or duplicated")
                patients[qid] = patient
                self.source_rows[qid] = semantic_hash(row)
        require(len(patients) == QUESTIONS, "Dataset must contain exactly 2,931 unique questions")
        require(set(self.ledger_snapshot["selections"]) == set(patients),
                "V3 is incomplete: acceptance ledger does not contain the exact 2,931-question cohort")
        canonical_rows = {}
        self.canonical_hash = self.remember(self.canonical)
        for row in iter_result_rows(self.canonical):
            qid = row.get("question_id")
            require(qid in patients and qid not in canonical_rows, "Canonical V3 identities are unknown or duplicated")
            require(row.get("patient_fhir_id") == patients[qid], "Canonical V3 patient identity changed")
            require(not row.get("attempt_history") and "agent_answer" in row, "Canonical V3 contains retry history or lacks answer field")
            canonical_rows[qid] = semantic_hash(row)
        require(set(canonical_rows) == set(patients), "V3 is incomplete: canonical results do not contain exactly 2,931 questions")
        self.identity_hash = self.remember(self.identity_path)
        expected_identity = self.manifest["run_identities"][ARM]
        require(read_json(self.identity_path) == expected_identity, "Canonical V3 identity differs from manifest")
        require(expected_identity.get("agent_strategy") == "multi_agent_v3", "Manifest does not pin V3 strategy")
        require(input_identity_hash(source) == expected_identity["input_hash"], "Full input identity hash changed")
        hashes = self.manifest.get("source_sha256") or {}
        require(set(REQUIRED_SOURCES) <= set(hashes), "Manifest lacks required frozen source pins")
        require(hashes["scripts/v3_full_run.py"] == APPROVED_CLASSIFIER["sha256"]
                and self.manifest.get("classifier_version") == APPROVED_CLASSIFIER["version"],
                "Classifier differs from the independently approved V3 operational amendment")
        for relative, expected in hashes.items():
            path = (ROOT / relative).resolve()
            require(path.is_relative_to(ROOT) and (path.suffix == ".py" or relative == "requirements.txt"),
                    "Unexpected source pin path")
            require(self.remember(path) == expected, "Frozen source changed")
        for key in ("families", "split_manifest"):
            entry = self.manifest[key]
            require(self.remember(entry["path"]) == entry["sha256"], "Frozen analysis input changed")
        for name in ("pricing.json", "split.json"):
            expected = (self.manifest.get("pinned_files_sha256") or {}).get(name)
            require(expected and self.remember(self.run / name) == expected, "Frozen pricing/split snapshot changed")
        classifier, version = load_classifier(ROOT / "scripts/v3_full_run.py")
        require(version == self.manifest.get("classifier_version"), "Classifier version differs from manifest")
        self.classifier_version = version
        selected, choices, retries = {}, {}, Counter()
        counts = {"raw_attempts": 0, "runtime_failures": 0, "intrinsic_errors": 0}
        batches = sorted((self.run / "attempts" / ARM).glob("round-*.batch.json"),
                         key=lambda path: int(path.name.split("-")[1].split(".")[0]))
        require(bool(batches), "V3 has no sealed source attempts")
        self.sealed_rounds, audit_owners = [], {}
        for batch_path in batches:
            seal_path = batch_path.with_suffix(".seal.json")
            require(seal_path.is_file(), "V3 still has an unsealed batch")
            batch, seal = read_json(batch_path), read_json(seal_path)
            number = int(batch_path.name.split("-")[1].split(".")[0])
            require(batch.get("arm") == ARM and batch.get("round") == number, "Batch ownership/round changed")
            require(self.remember(batch_path) == seal["batch_sha256"], "Batch seal mismatch")
            self.remember(seal_path)
            output, shard = Path(batch["output"]).resolve(), Path(batch["input"]).resolve()
            require(output.is_relative_to(self.run) and shard.is_relative_to(self.run), "Attempt path escapes run directory")
            require(self.remember(shard) == batch["input_sha256"], "Shard input hash changed")
            shard_ids = []
            with shard.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    qid = row["question_id"]
                    require(qid in patients and semantic_hash(row) == self.source_rows[qid], "Shard input differs from authoritative dataset")
                    shard_ids.append(qid)
            require(len(set(shard_ids)) == len(shard_ids) and shard_ids == batch["question_ids"], "Shard question IDs changed")
            self.sealed_rounds.append(number)
            if not output.exists():
                require(seal.get("result_sha256") is None and seal.get("identity_sha256") is None, "Missing sealed result")
                continue
            identity_path = Path(str(output) + ".run.json")
            require(self.remember(output) == seal["result_sha256"], "Raw result seal mismatch")
            require(self.remember(identity_path) == seal["identity_sha256"], "Raw identity seal mismatch")
            identity = read_json(identity_path)
            require(identity == batch["run_identity"] and set(identity) == set(expected_identity), "Raw attempt identity changed")
            require(all(value == expected_identity[key] for key, value in identity.items() if key != "input_hash"),
                    "Raw attempt model/settings/inference identity differs from manifest")
            require(identity["input_hash"] == input_identity_hash(shard), "Shard identity is not bound to shard contents")
            seen = set()
            for row in iter_result_rows(output):
                qid = row.get("question_id")
                require(qid in shard_ids and qid not in seen and row.get("patient_fhir_id") == patients[qid], "Raw question/patient ownership changed")
                seen.add(qid)
                ref = row.get("audit_artifacts")
                if ref is not None:
                    require(isinstance(ref, dict) and ref.get("version") == 2 and ref.get("codec") == "json+gzip"
                            and ref.get("question_id") == qid and ARTIFACT_PATTERN.fullmatch(ref.get("path", "")), "Raw audit descriptor ownership is invalid")
                    audit = (output.parent / ref["path"]).resolve()
                    require(audit.is_relative_to(output.parent), "Raw audit path escapes owning directory")
                    raw_audit_hash = self.remember(audit)
                    require(audit not in audit_owners or audit_owners[audit] == qid, "Audit reused across question owners")
                    audit_owners[audit] = qid
                # Restoring checks the uncompressed SHA256 and payload question owner.
                full = restore_result_audit(row, output.parent, restore_history=False)
                decision = classifier(full, None)
                if not attempted(row):
                    require(decision["status"] == "pending", "Unattempted placeholder classified as terminal")
                    continue
                require(qid not in selected, "An already accepted question was rerun")
                counts["raw_attempts"] += 1
                if decision["status"] == "runtime":
                    retries[qid] += 1
                    counts["runtime_failures"] += 1
                elif decision["status"] == "accepted":
                    require(ref is not None, "Accepted attempt lacks owned audit artifact")
                    compact = {key: value for key, value in row.items() if key != "attempt_history"}
                    selected[qid] = semantic_hash(compact)
                    choices[qid] = {"attempt_file": str(output.relative_to(self.run)), "round": number,
                                    "runtime_failures_before": retries[qid], "selection": decision["reason"], "runtime_clean": True}
                    counts["intrinsic_errors"] += int(bool(row.get("error")))
                    require(self.remember(self.run / ref["path"]) == raw_audit_hash, "Canonical audit differs from accepted raw audit")
                else:
                    raise ValueError("A sealed V3 attempt has unresolved attention or invalid classification")
            require(seen == set(shard_ids), "Raw result omitted scheduled placeholders")
        require(selected == canonical_rows, "Canonical V3 differs from first-clean raw selection")
        require(choices == self.ledger_snapshot["selections"], "V3 ledger differs from first-clean source-attempt selection")
        require(counts == self.ledger_snapshot["counts"], "V3 operational counts differ from sealed replay")
        self.replay_counts, self.audit_count = counts, len(audit_owners)
        self.selected_row_digest = semantic_hash(selected)
        self.check_stable()
        return self

    def check_stable(self):
        require(sha256(self.canonical) == self.canonical_hash, "V3 canonical changed during delivery")
        require(sha256(self.identity_path) == self.identity_hash, "V3 identity changed during delivery")
        require(sha256(self.manifest_path) == self.manifest_hash, "Run manifest changed during delivery")
        latest = read_json(self.ledger_path)
        require(semantic_hash(v3_ledger(latest)) == self.ledger_slice_hash, "V3 acceptance slice changed during delivery")
        return latest

    def verify_inventory(self):
        for path, metadata in self.inventory.items():
            require(sha256(path) == metadata["sha256"], "Verified immutable source changed during export")

    def snapshot(self, destination):
        destination.mkdir()
        # All hardlinked run artifacts are sealed or atomically replaced by the
        # supervisor. No source path is written; links avoid duplicating audits.
        for original, metadata in self.inventory.items():
            if not original.is_relative_to(self.run):
                continue
            relative = original.relative_to(self.run)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            require(sha256(original) == metadata["sha256"], "Verified artifact changed before snapshot")
            if (relative.as_posix() in {"run_manifest.json", "pricing.json", "split.json"}
                    or "audit" not in relative.parts and metadata["bytes"] <= 1024 * 1024):
                shutil.copyfile(original, target)
                continue
            try:
                os.link(original, target)
            except OSError:
                shutil.copyfile(original, target)
            require(sha256(target) == metadata["sha256"], "Snapshot artifact hash mismatch")
        (destination / "run_manifest.json").write_bytes(self.manifest_bytes)
        (destination / "acceptance.json").write_bytes(self.ledger_bytes)
        state_path = self.run / "supervisor_state.json"
        if state_path.exists():
            shutil.copyfile(state_path, destination / state_path.name)
        operations = self.run / "operations.jsonl"
        if operations.exists():
            raw = operations.read_bytes()
            complete = raw[:raw.rfind(b"\n") + 1] if b"\n" in raw else b""
            for line in complete.splitlines():
                json.loads(line)
            (destination / operations.name).write_bytes(complete)
        self.check_stable()


def load_analyzer():
    return importlib.import_module("scripts.v3_full_analysis")


def file_manifest(directory):
    return {str(path.relative_to(directory)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(directory.rglob("*")) if path.is_file()}


def deliver(run_dir, output_dir):
    run, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    require(not output.exists(), "Output directory must be NEW")
    require(not output.is_relative_to(run), "Delivery output must be outside the live run directory")
    require(not run.is_relative_to(output), "Delivery output must not contain the live run directory")
    gate = CompleteV3Gate(run).validate()  # No analyzer/scorer import before this succeeds.
    output.mkdir(parents=True)
    snapshot = output / "validated_v3_input_snapshot"
    gate.snapshot(snapshot)
    gate.verify_inventory()
    gate.check_stable()
    analyzer = load_analyzer()
    report, records = analyzer.analyze(snapshot, partial=True, n_boot=10000)
    require(report["status"] == "PARTIAL_PROGRESS_DO_NOT_REPORT_AS_FINAL" and report["partial_mode"] is True,
            "Frozen global partial status was not preserved")
    require(report["completeness"][ARM] == {"expected": QUESTIONS, "accepted": QUESTIONS, "missing": 0,
                                         "missing_question_ids": [], "complete": True}, "Analyzed V3 cohort incomplete")
    require(len(records) == QUESTIONS and all(row["arm"] == ARM and row["runtime_clean_selection_verified"] for row in records),
            "Analysis included partial controls or unverified selections")
    analyzer.write_report(report, records, output / "global_partial")
    # Project the frozen statistics; no scoring, metric or confidence-interval
    # calculation is reimplemented here.
    projected = {key: value for key, value in report.items() if key not in
                 {"all_arms_accepted_metrics", "comparisons", "arms", "groups", "mechanisms", "phase_usage",
                  "completeness", "accepted_results", "operational_context"}}
    projected.update(status=STATUS, benchmark_complete=False, v3_complete=True, scope=ARM,
                     arms={ARM: report["arms"][ARM]}, completeness={ARM: report["completeness"][ARM]},
                     accepted_results={ARM: report["accepted_results"][ARM]},
                     groups={key: [row for row in rows if row["arm"] == ARM] for key, rows in report["groups"].items()},
                     mechanisms={ARM: report["mechanisms"][ARM]},
                     phase_usage=[row for row in report["phase_usage"] if row["arm"] == ARM], comparisons=[],
                     operational_context={"per_arm": {ARM: report["operational_context"]["per_arm"][ARM]},
                                          "note": "V3 operational wall time includes runtime retries; accepted latency/cost exclude them."})
    analyzer.write_report(projected, records, output / "v3_complete")
    (output / "v3_complete" / "paired_comparisons.csv").unlink()
    latest = gate.check_stable()
    # Compare all immutable provenance again, allowing whole-ledger/control updates.
    gate.verify_inventory()
    metrics = report["arms"][ARM]["metrics"]
    provenance = {str(path): metadata for path, metadata in gate.inventory.items()}
    (output / "verified_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    receipt = {"schema_version": 1, "status": STATUS, "benchmark_complete": False, "v3_complete": True,
               "generated_at": datetime.now(timezone.utc).isoformat(), "scope": ARM, "questions": QUESTIONS,
               "live_run_directory": str(run), "output_directory": str(output),
               "source_manifest_sha256": gate.manifest_hash, "v3_canonical_sha256_before": gate.canonical_hash,
               "v3_canonical_sha256_after": sha256(gate.canonical), "v3_identity_sha256": gate.identity_hash,
               "v3_ledger_slice_sha256_before": gate.ledger_slice_hash,
               "v3_ledger_slice_sha256_after": semantic_hash(v3_ledger(latest)),
               "whole_ledger_sha256_at_gate": gate.initial_ledger_hash,
               "whole_ledger_sha256_after": sha256(gate.ledger_path),
               "classifier_version": gate.classifier_version, "sealed_v3_rounds": gate.sealed_rounds,
               "independently_approved_classifier": APPROVED_CLASSIFIER,
               "raw_v3_attempt_counts": gate.replay_counts, "verified_raw_audits": gate.audit_count,
               "selected_raw_rows_sha256": gate.selected_row_digest,
               "missing_measurements": {key: value["missing_n"] for key, value in metrics.items()},
               "controls": {"status": "PENDING_MATCHED_FINAL_ANALYSIS", "included_in_delivery": False,
                            "accepted_counts_at_gate": gate.initial_control_counts,
                            "accepted_counts_after": {arm: len((latest.get("arms") or {}).get(arm) or {})
                                                       for arm in ("direct_llm", "single_agent")}},
               "global_partial_export": {"directory": "global_partial", "status": report["status"],
                                         "note": "Frozen exporter output on a V3-only snapshot. Control artifacts intentionally omitted; these zero snapshot counts are not live control counts."},
               "snapshot": {"directory": "validated_v3_input_snapshot", "self_contained_or_portable": False,
                            "original_manifest_bytes_preserved": True, "original_ledger_bytes_preserved": True,
                            "omitted_live_artifacts": ["Control accepted results/identities, raw attempts and audits", "Provider/runtime/configuration/credential files"],
                            "operational_log": "Complete JSONL prefix captured at snapshot time",
                            "storage": "Small metadata is copied. Verified audits and larger immutable V3 result files may be hardlinks to live inodes; do not edit snapshot files or change their permissions. Use v3_complete tables for analysis.",
                            "absolute_references": "Original dataset, family, offline FHIR and batch input/output paths remain absolute references to verified live immutable artifacts."},
               "frozen_full_benchmark_final_gate_changed": False,
               "scoring": report["scoring"], "statistics": report["statistics"], "pricing": report["pricing"],
               "helper_sha256": sha256(Path(__file__)),
               "guarantees": ["Full V3 cohort and first-clean provenance verified before any scoring",
                              "No partial control answers were scored or presented",
                              "Frozen analyzer/statistics used unchanged; global partial output preserved",
                              "Live source, runtime, results, ledger, processes and scheduling unchanged",
                              "The complete matched three-arm benchmark and its final analysis are still pending"]}
    (output / "V3_COMPLETE_CONTROLS_PENDING.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    (output / "README.txt").write_text(
        "V3_COMPLETE_CONTROLS_PENDING\n\nThe V3 arm contains all 2,931 validated first-runtime-clean outcomes. "
        "Use v3_complete/ for its per-question data, aggregates, evidence and revision diagnostics. "
        "Matched controls and the full three-arm final analysis remain pending.\n\n"
        "global_partial/ preserves the frozen exporter's globally partial output from a detached V3-only snapshot. "
        "Its absent controls do not describe live control progress. See the receipt for observed live control counts.\n"
        "Treat validated_v3_input_snapshot as read-only: its larger results/audits can be hardlinks to the run. "
        "Editing or chmod of snapshot hardlinks would also affect their source inode. Absolute source references remain; this is not a portable archive.\n"
        "No cross-arm comparison or combined-arm total is included in the V3 delivery. "
        "Estimated cost uses the frozen official OpenAI schedule under the user's Azure rate assumption.\n")
    # Data artifacts are fingerprinted separately from their hash manifest.
    (output / "delivery_artifact_hashes.json").write_text(json.dumps(file_manifest(output), indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        receipt = deliver(args.run_dir, args.output_dir)
    except ValueError as error:
        print("Complete V3 delivery refused: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps({"status": receipt["status"], "questions": receipt["questions"],
                      "benchmark_complete": False, "receipt": str(args.output_dir / "V3_COMPLETE_CONTROLS_PENDING.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
