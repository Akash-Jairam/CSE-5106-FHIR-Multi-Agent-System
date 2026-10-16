"""Freeze a substantial design experiment without reading credentials.

Create one record before an entire fixed-subset/full run. Small development
checks intentionally do not use this journal. Raw outputs remain local;
source snapshots, checksums, rationale and compact results are reviewable.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("agent", "tools", "utils", "scripts", "tests", "docs", "manifests")
ROOT_FILES = ("run_agent.py", "fhir_client.py", "requirements.txt", "config.py", "config.example.yml", "README.md", "AGENTS.md")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def freeze(args):
    destination = Path(args.record)
    if (destination / "manifest.json").exists():
        raise ValueError("This experiment is already frozen; choose a distinct record directory")
    source = [ROOT / name for name in ROOT_FILES if (ROOT / name).is_file()]
    for directory in SOURCE_DIRS:
        source.extend(path for path in (ROOT / directory).rglob("*")
                      if path.is_file() and path.suffix in {".py", ".md", ".txt", ".json", ".csv", ".yml", ".yaml"}
                      and "__pycache__" not in path.parts)
    source = sorted(set(source))
    destination.mkdir(parents=True, exist_ok=True)
    description = Path(args.description).read_text()
    (destination / "design.txt").write_text(description)
    archive = destination / "source.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as snapshot:
        for path in source:
            snapshot.write(path, str(path.relative_to(ROOT)))
    git = lambda *command: subprocess.check_output(["git", *command], cwd=ROOT, text=True).strip()
    manifest = {"version": 1, "experiment_id": destination.name,
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "branch": git("branch", "--show-current"), "code_commit": git("rev-parse", "HEAD"),
                "working_tree_status": git("status", "--short"),
                "design_sha256": digest(destination / "design.txt"),
                "source_archive_sha256": digest(archive),
                "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in source},
                "dataset_sha256": digest(ROOT / "final_dataset/questions_answers_sql_fhir.csv"),
                "split_sha256": digest(ROOT / "manifests/split.json"),
                "output_directory": str(Path(args.output_directory).resolve()),
                "model": args.model, "max_total_tokens": args.max_total_tokens,
                "max_tool_calls": args.max_tool_calls, "estimated_input_guard": 32000,
                "expected_questions_per_arm": args.questions, "arms": args.arm,
                "command": args.command_text,
                "scoring": "Unchanged typed scorer; all scheduled first attempts; terminal errors incorrect",
                "selection_note": "The fixed300 is reused for architecture development; its gains are exploratory.",
                "status": "frozen_before_large_run"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen {destination.name}: {len(source)} source files; {args.questions} questions per arm")


def finish(args):
    destination = Path(args.record)
    manifest = json.loads((destination / "manifest.json").read_text())
    outputs = Path(manifest["output_directory"])
    results = {}
    for arm in manifest["arms"]:
        path = outputs / (arm + ".json")
        sidecar = Path(str(path) + ".run.json")
        if not path.is_file() or not sidecar.is_file():
            raise ValueError(f"Missing output or run identity for {arm}")
        results[arm] = {"result_sha256": digest(path),
                        "run_identity": json.loads(sidecar.read_text()),
                        "result_path": str(path)}
    provider = outputs / ".local-provider.json"
    result_manifest = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "arms": results,
                       "provider_binding": json.loads(provider.read_text()) if provider.exists() else None}
    if args.report:
        report = Path(args.report)
        result_manifest["report_sha256"] = digest(report)
        (destination / "results.json").write_bytes(report.read_bytes())
    (destination / "outcome_manifest.json").write_text(json.dumps(result_manifest, indent=2) + "\n")
    print(f"Recorded checksums and outcomes for {destination.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["freeze", "finish"])
    parser.add_argument("--record", required=True)
    parser.add_argument("--description")
    parser.add_argument("--output_directory")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--max_total_tokens", type=int, default=200000)
    parser.add_argument("--max_tool_calls", type=int, default=30)
    parser.add_argument("--questions", type=int, default=300)
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--command_text")
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.operation == "freeze":
        if not all([args.description, args.output_directory, args.arm, args.command_text]):
            parser.error("freeze needs description, output_directory, arm and command_text")
        freeze(args)
    else:
        finish(args)


if __name__ == "__main__":
    main()
