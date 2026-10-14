"""Verify the frozen run without inference or extraction of its archive."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def digest_file(path):
    with path.open("rb") as handle:
        return digest_stream(handle)


def digest_stream(handle):
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Relocated raw-result archive")
    args = parser.parse_args()
    snapshot = Path(__file__).resolve().parent
    root = snapshot.parent.parent
    manifest = json.loads((snapshot / "snapshot_manifest.json").read_text())
    inventory = json.loads((snapshot / "baseline_inventory.json").read_text())
    source = json.loads((snapshot / "source_sha256.json").read_text())

    for line in (snapshot / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = (snapshot / relative).resolve()
        if not path.is_relative_to(snapshot):
            raise ValueError("Invalid snapshot checksum path")
        if digest_file(path) != expected:
            raise ValueError(f"Copied artifact changed: {relative}")

    ref = f"refs/tags/{manifest['tag']}"
    for relative, expected in source.items():
        content = subprocess.check_output(
            ["git", "show", f"{ref}:{relative}"], cwd=root
        )
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Tagged source changed: {relative}")

    archive = args.archive or root / manifest["archive"]["path"]
    if digest_file(archive) != manifest["archive"]["sha256"]:
        raise ValueError("Raw archive checksum does not match")
    prefix = manifest["archive"]["prefix"]
    expected_members = {
        f"{prefix}/{item['path']}": item for item in inventory["files"]
    }
    seen = set()
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            if not member.isfile() or member.name not in expected_members:
                raise ValueError(f"Unexpected archive member: {member.name}")
            if member.name in seen:
                raise ValueError(f"Duplicate archive member: {member.name}")
            expected = expected_members[member.name]
            if member.size != expected["size"]:
                raise ValueError(f"Archive member size changed: {member.name}")
            with bundle.extractfile(member) as handle:
                if digest_stream(handle) != expected["sha256"]:
                    raise ValueError(f"Archive member changed: {member.name}")
            seen.add(member.name)
    if seen != set(expected_members):
        raise ValueError("Archive is missing original run files")
    print(f"Verified {len(source)} source files at tag {manifest['tag']}, "
          f"copied artifacts, archive checksum, and all {len(seen)} original files.")


if __name__ == "__main__":
    main()
