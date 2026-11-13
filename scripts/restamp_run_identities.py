"""Re-stamp saved run identities after a runtime change that cannot alter their results.

A result file's .run.json binds it to the code fingerprint (run_agent.code_fingerprint)
and, for recovery arms, to the perturbation manifest's bytes. Commit 4ecbe0d
taught tools/perturbation.py to also load version-3 (multi-record) manifests;
version-2 manifests load and omit exactly as before, and manifests/perturbations.json
was rebuilt with identical entries. That marked every earlier result stale.

This script moves a saved identity from the old fingerprint to the current
one only when all of these hold, and otherwise leaves the file alone:
  - the fingerprinted files (run_agent.py, fhir_client.py, requirements.txt,
    agent/, tools/, utils/) differ from --from-commit only in --allow paths,
    and have no uncommitted changes;
  - the saved code_hash equals --from-commit's fingerprint computed in this
    environment (same package versions), so the run used exactly that code;
  - a saved perturbations_hash is updated only if it is the hash of
    --from-commit's perturbation manifest and the current manifest has the
    same entries.
Run it where the results were produced. It prints a dry run unless --apply
is given, keeps a .bak copy of each changed file, and appends what it changed
to restamp_log.json in each directory.

Usage:
    python scripts/restamp_run_identities.py output/experiments/qwen3:32b/held_out [--apply]
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import run_agent  # noqa: E402

DEFAULT_FROM = "4ecbe0d~1"
DEFAULT_ALLOW = ("tools/perturbation.py",)
FINGERPRINTED = ("run_agent.py", "fhir_client.py", "requirements.txt", "agent", "tools", "utils")
PERTURBATIONS = "manifests/perturbations.json"


def git(*args, binary=False):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True)
    return result.stdout if binary else result.stdout.decode()


def fingerprint_at(commit):
    """run_agent.code_fingerprint() of `commit`'s tree, with this environment's package versions."""
    names = git("ls-tree", "-r", "--name-only", commit).split()
    paths = [n for n in names if n in FINGERPRINTED[:3]
             or (n.split("/")[0] in FINGERPRINTED[3:] and n.endswith(".py"))]
    digest = hashlib.sha256()
    for name in sorted(paths, key=Path):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(git("show", f"{commit}:{name}", binary=True))
        digest.update(b"\0")
    digest.update(json.dumps(run_agent.runtime_dependency_versions(ROOT / "requirements.txt"), sort_keys=True).encode())
    return digest.hexdigest()


def check_code_change(commit, allow):
    changed = set(git("diff", "--name-only", commit, "HEAD", "--", *FINGERPRINTED).split())
    if changed - set(allow):
        raise SystemExit(f"Fingerprinted files changed beyond {sorted(allow)}: {sorted(changed - set(allow))}")
    if git("status", "--porcelain", "--", *FINGERPRINTED).strip():
        raise SystemExit("Fingerprinted files have uncommitted changes; commit or stash them first")
    return sorted(changed)


def perturbation_hashes(commit):
    """(old manifest hash, current manifest hash) if their entries are identical, else None."""
    old = git("show", f"{commit}:{PERTURBATIONS}", binary=True)
    new = (ROOT / PERTURBATIONS).read_bytes()
    if json.loads(old)["perturbations"] != json.loads(new)["perturbations"]:
        return None
    return hashlib.sha256(old).hexdigest(), hashlib.sha256(new).hexdigest()


def plan(directory, old_code, new_code, hashes):
    """[(path, {field: (old, new)})] for the identities in `directory` that can be re-stamped."""
    changes, skipped = [], []
    for path in sorted(Path(directory).rglob("*.run.json")):
        identity = json.loads(path.read_text(encoding="utf-8"))
        if identity.get("code_hash") == new_code:
            continue
        if identity.get("code_hash") != old_code:
            skipped.append(path)
            continue
        fields = {"code_hash": (old_code, new_code)}
        saved = identity.get("perturbations_hash")
        if saved is not None:
            if hashes is None or saved != hashes[0]:
                skipped.append(path)
                continue
            fields["perturbations_hash"] = hashes
        changes.append((path, fields))
    return changes, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--from-commit", default=DEFAULT_FROM)
    parser.add_argument("--allow", nargs="+", default=list(DEFAULT_ALLOW))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    changed = check_code_change(args.from_commit, args.allow)
    old_code, new_code = fingerprint_at(args.from_commit), run_agent.code_fingerprint()
    hashes = perturbation_hashes(args.from_commit)
    print(f"code {old_code[:12]} -> {new_code[:12]} (changed: {', '.join(changed) or 'nothing'}); perturbation "
          f"manifest {'entries identical' if hashes else 'entries differ: recovery identities are left alone'}")
    head = git("rev-parse", "HEAD").strip()
    for directory in args.directories:
        changes, skipped = plan(directory, old_code, new_code, hashes)
        print(f"\n{directory}: {len(changes)} to re-stamp, {len(skipped)} from other code or manifests left alone")
        for path in skipped:
            print(f"  left alone  {path.name}")
        for path, fields in changes:
            print(f"  {'re-stamped' if args.apply else 'would stamp'} {path.name}: {', '.join(fields)}")
            if not args.apply:
                continue
            identity = json.loads(path.read_text(encoding="utf-8"))
            shutil.copy2(path, str(path) + ".bak")
            identity.update({field: new for field, (_, new) in fields.items()})
            run_agent._write_json(identity, str(path))
        if args.apply and changes:
            log_path = Path(directory) / "restamp_log.json"
            log = json.loads(log_path.read_text()) if log_path.exists() else []
            log.append({"time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                        "from_commit": git("rev-parse", args.from_commit).strip(), "to_commit": head,
                        "changed_files": changed,
                        "files": {str(p.relative_to(directory)): {f: list(v) for f, v in fields.items()}
                                  for p, fields in changes}})
            log_path.write_text(json.dumps(log, indent=1) + "\n")
    if not args.apply:
        print("\nDry run; pass --apply to write.")


if __name__ == "__main__":
    main()
