"""Check every leaf in the consolidated results manifest without inference."""
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parents[1]
repo = base.parents[1]
manifest = json.loads((base / 'PACKAGE_MANIFEST.json').read_text())
checked = total_bytes = 0
for item in manifest['files']:
    path = (repo / item['path']).resolve()
    if not path.is_relative_to(repo) or not path.is_file():
        raise ValueError('Missing or escaping package path: '+item['path'])
    if path.stat().st_size != item['bytes']:
        raise ValueError('Size differs: '+item['path'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != item['sha256']:
        raise ValueError('SHA256 differs: '+item['path'])
    checked += 1
    total_bytes += item['bytes']
if checked != manifest['file_count'] or total_bytes != manifest['total_bytes']:
    raise ValueError('Aggregate inventory differs')
print(json.dumps({'status':'PASS','files':checked,'bytes':total_bytes,
                  'scope':'Manifest leaves; manifest excludes itself.',
                  'provider_calls':False}, indent=2))
