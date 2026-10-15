"""Pure independent V8 integrity tests; no runtime/core/scorer/QA/config imports."""
from pathlib import Path
import hashlib,json,os,sys,unittest
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
os.environ['FHIR_LOCAL_AZURE_CONFIG']=''
os.environ['FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR']=''
OUT=Path(__file__).resolve().parent
files=['tests/test_v8_study_integrity.py','scripts/v8_study_integrity.py','scripts/audit_store.py',
       'benchmarks/design-v7-20261015/integrity/v7_guard.py',
       'benchmarks/design-v7-20261015/integrity/reviewed_v4_guard.py']
def hashes():return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
before=hashes()
with (OUT/'independent_integrity_tests.log').open('w') as stream:
 suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_v8_study_integrity')
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
after=hashes();passed=result.wasSuccessful() and before==after
review={'kind':'independent_V8_completion_integrity_synthetic_review',
 'recorded_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS' if passed else 'FAIL',
 'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
 'skips':len(result.skipped),'source_stable_during_tests':before==after,'source_sha256':after,
 'owned_edits':['tests/test_v8_study_integrity.py'],
 'scope':['Handwritten eligible and prose-only candidate envelopes with independent canonical hashes and explicit execution/source-check traces.',
 'Four policy case,existing audit_store gzip externalization/restoration,sharedA conceptual prefix and physicalAonce,question-owner/hash/corruptgzip/hardlinkedcache controls.',
 'Stale source/execution/workspace,own-call bindings,diagnostic/counter contradictions,unknown/fatal/overshoot accounting.',
 'Synthetic safe SDK start/end pairs,known cached usage,unknown usage,missing/duplicate/pending/partial events,unrecorded known calls and extra sends.',
 'Observed owned terminal attestation gates before provenance;479cannot reach case/journal validation even if metadata falsely claims480.'],
 'finding_resolved':'Known nonfatal cached field now rejects boolean/noninteger/negative or cached>prompt; legitimate absent cache usage remains a separate unavailable metadata issue.',
 'limits':['Pure stdlib plus reviewed audit-store/shared integrity helpers; no core,model,scorer,QA,configuration,network or live outcomes were accessed.',
 'These fixtures test retained integrity/provenance boundaries,not model efficacy,clinical semantics or actual FHIR completeness.',
 'Full480 source/runtime pin/archive roundtrip is not synthesized here; root owns formal freeze and actual postterminal deep check.',
 'Fatal pre-send/observer gaps are reported without zero-usage imputation; known nonfatal SDK coverage must reconcile.']}
(OUT/'independent_integrity_review.json').write_text(json.dumps(review,indent=2,sort_keys=True)+'\n')
names=['run_independent_integrity_tests.py','independent_integrity_command.txt','independent_integrity_tests.log','independent_integrity_review.json']
delivery={'status':review['status'],'tests_run':result.testsRun,'source_stable':before==after,
 'files':{name:{'sha256':hashlib.sha256((OUT/name).read_bytes()).hexdigest(),'bytes':(OUT/name).stat().st_size} for name in names}}
(OUT/'independent_integrity_delivery.json').write_text(json.dumps(delivery,indent=2,sort_keys=True)+'\n')
print(json.dumps(delivery,indent=2))
raise SystemExit(not passed)
