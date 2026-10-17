"""Inventory the finalized results package. Run only after all writers finish."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

base = Path(__file__).resolve().parents[1]
repo = base.parents[1]
target = base / 'PACKAGE_MANIFEST.json'
prefixes = [
    'results/v3',
    'benchmarks/design-v3d-20261014',
    'benchmarks/v3-full-run-20261015/completed_v3',
    'benchmarks/v3-full-run-20261015/cache_metadata_reconciliation_20261017',
]
files = []
for prefix in prefixes:
    directory = repo / prefix
    if not directory.is_dir():
        raise ValueError('Required package directory is missing: '+prefix)
    for p in sorted(directory.rglob('*')):
        if not p.is_file() or p == target or '__pycache__' in p.parts or p.name == '.DS_Store':
            continue
        if p.suffix in {'.pyc', '.tmp', '.partial'}:
            raise ValueError('Unfinished or unintended package file: '+str(p.relative_to(repo)))
        n = p.stat().st_size
        if n >= 90_000_000:
            raise ValueError('Package blob exceeds 90MB delivery gate: '+str(p.relative_to(repo)))
        digest = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        files.append({'path':str(p.relative_to(repo)), 'bytes':n, 'sha256':digest.hexdigest()})
files.sort(key=lambda x:x['path'])
manifest = {
    'schema_version':1, 'status':'CONSOLIDATED_RESULTS_DELIVERY',
    'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'branch':'results/v3-ful-run',
    'source_base_commit':subprocess.check_output(['git','rev-parse','f71a8d9'],cwd=repo,text=True).strip(),
    'cohort_scope':{'development_single':300,'development_v3':300,'full_v3_overall':2931,
                    'full_single_available':2929,'primary_matched_pairs':2929,'full_patients':94},
    'run_registry':'results/v3/RUN_REGISTRY.json', 'protocol':'results/v3/METHODS.txt',
    'editable_analysis':'results/v3/ANALYSIS.docx',
    'inventory_prefixes':prefixes,
    'self_excluded':True,
    'file_count':len(files), 'total_bytes':sum(f['bytes'] for f in files),
    'largest_blob_bytes':max(f['bytes'] for f in files),
    'runtime_implementation_merged':False, 'new_inference':False,
    'validation_receipts':[
       'results/v3/internal/validation/consolidation_review.json',
       'results/v3/internal/validation/document_quality_review.json',
       'results/v3/development/validation.json',
       'results/v3/full_single_agent/validation.json',
       'results/v3/full_single_agent/internal/portable_archive_validation.json',
       'results/v3/full_v3/PORTABLE_VERIFICATION.json',
    ],
    'preservation_note':'Historical paths retain original labels. Main reporting names the accepted architecture V3. Raw archive manifests define byte/semantic preservation. Original full completion gates are unchanged.',
    'files':files,
}
target.write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ['status','file_count','total_bytes','largest_blob_bytes']}))
