"""Package only the frozen completion report's explicit public archive allowlist."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[3]
RECORD = ROOT / 'benchmarks/design-v8-protocol-repair-20261015'
RAW = ROOT / 'output/experiments/gpt-6-luna/v8_mechanism_120_protocol_repair_20261015'
GUARD_SHA = '677aaba828fd29b0546a5b6fc7ed7b1faa691a8d1b6f2e602c8c5113433613d7'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    report_path = RECORD / 'completion_integrity.json'
    assert digest(report_path) == GUARD_SHA
    report = json.loads(report_path.read_text())
    assert report['status'] == 'PASS' and report['policy_outcomes'] == 480
    names = report['archive_allowlist']
    assert len(names) == len(set(names)) == 506
    archive = ROOT / 'output/archives/v8_mechanism_120_protocol_repair_20261015.completed.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    members = []
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for name in names:
            path = RAW / name
            assert not path.is_symlink() and path.is_file()
            assert path.resolve().is_relative_to(RAW.resolve())
            assert not name.startswith('case_cache/')
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            sha = hashlib.sha256()
            size = 0
            with path.open('rb') as source, target.open(info, 'w', force_zip64=True) as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    sha.update(chunk)
                    size += len(chunk)
                    output.write(chunk)
            assert sha.hexdigest() == report['raw_sha256'][name], name
            members.append({'path': name, 'sha256': sha.hexdigest(), 'bytes': size})
    with zipfile.ZipFile(archive) as saved:
        assert saved.namelist() == names
        for member in members:
            with saved.open(member['path']) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == member['sha256']
    assert digest(report_path) == GUARD_SHA
    manifest = {
        'kind': 'v8_completed_raw_local_archive',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'completion_integrity_sha256': GUARD_SHA,
        'archive_path': str(archive.relative_to(ROOT)),
        'archive_sha256': digest(archive),
        'archive_bytes': archive.stat().st_size,
        'members': members,
        'member_count': len(members),
        'uncompressed_bytes': sum(item['bytes'] for item in members),
        'roundtrip_all_member_sha256_verified': True,
        'scope': 'Exact completion archive_allowlist; raw archive remains ignored and local, while its manifest is committed.',
        'excluded': ['case_cache and pickle files', 'raw launch log', 'configuration or credentials', 'files absent from the completion archive_allowlist'],
    }
    destination = RECORD / 'raw_archive_manifest.json'
    with destination.open('x') as output:
        output.write(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({key: manifest[key] for key in ('archive_path', 'archive_sha256', 'archive_bytes', 'member_count', 'roundtrip_all_member_sha256_verified')}))


if __name__ == '__main__':
    main()
