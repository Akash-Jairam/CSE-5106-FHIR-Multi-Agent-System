"""Authorized offline independent V8 check; no credential discovery or inference."""
from pathlib import Path
import hashlib
import json
import os
import sys
import unittest
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
os.environ['FHIR_LOCAL_AZURE_CONFIG']=''
os.environ['FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR']=''
OUT=Path(__file__).resolve().parent
names=['tests/test_fhir_v8_investigator.py','agent/counterexample_investigator_agent.py',
       'agent/counterexample_investigator_phase.py','scripts/v8_mechanism_study.py','scripts/v8_study_cache.py',
       'agent/single_agent.py','agent/base_agent.py','agent/continuous_primary.py',
       'agent/residual_trajectory_agent.py','agent/residual_trajectory_phase.py',
       'tools/python_executor.py','tools/python_worker.py','tools/field_check.py','tools/cache.py',
       'utils/core_utils.py','utils/answer_scorer.py','docs/pre_v8_review_20261015/proposed_design_and_tests.txt']
def source_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
before=source_hashes()
with (OUT/'independent_tests.log').open('w') as stream:
    suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_fhir_v8_investigator')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
after=source_hashes()
passed=result.wasSuccessful() and before==after
review={'kind':'independent_V8_synthetic_integration_review','recorded_utc':datetime.now(timezone.utc).isoformat(),
 'status':'PASS' if passed else 'FAIL','tests_run':result.testsRun,'failures':len(result.failures),
 'errors':len(result.errors),'skips':len(result.skipped),'source_stable_during_tests':before==after,
 'source_sha256':after,'scope':['Handwritten semantic expectations: tied extrema/permutation,projection,grouped aggregate,time bounds,relative day window,links,units,empty versus failed/missing data.',
 'Actual new core with scripted native LiteLLM message objects,real isolated Python workers,and original field checker over a mocked synthetic source store.',
 'Full200k/30tools/30SDK primary; residual auxiliary<=60k/6tools/6SDK; exact6 path and seventh-send refusal; known fallback; unknown and overshoot fatal even after correctA.',
 'Literal obligation and real observation lineage; invalid witness and judge substitutions; large observation refusal; source-withheld/proseA skip; staleB after retrieval; unadvertised tool refusal.',
 'CorrectA disagreement controls and harmful dependent calculation: fallible judge can preserve or wrongly replace a source-bound answer; host never establishes semantic proof.',
 'One intact real-tokenizer feasibility fixture,not a population coverage estimate.',
 'Cold private A cache,immutable postA bytes,independent physical WRC copies,exception restoration and no crosswarming; one physical A and recorder-only references.',
 'Pure synthetic120/480 study endpoints; no scoring before terminal completeness; H-M-F=actualR-A; predeclared gate and bootstrap; actual isolated CLI analysis with unchanged scorer; missing480 blocked; template retained recorder-side and SQL excluded.'],
 'limits':['Scripted responses establish contracts/accounting/lineage,not model counterexample discovery or empirical efficacy.',
 'No paid model calls,network,FHIR service,QA dataset,protected question/reference,outcome or secret configuration was accessed.',
 'Source checks validate exact source values/lineage; they do not prove semantics or population completeness.',
 'The nested project sandbox requires a worker-capable outer invocation; managed-sandbox-only invocation returns worker71.',
 'Original fullgoal design is subject to real unscored interface validation,prospective120 completion,deep integrity and frozen gates.'],
 'owned_edits':['tests/test_fhir_v8_investigator.py'],
 'command_file':str(OUT/'independent_command.txt'),'log_file':str(OUT/'independent_tests.log')}
(OUT/'independent_review.json').write_text(json.dumps(review,indent=2,sort_keys=True)+'\n')
files=['run_independent_v8_tests.py','independent_command.txt','independent_tests.log','independent_review.json']
delivery={'status':review['status'],'tests_run':result.testsRun,'source_stable':before==after,
 'files':{name:{'sha256':hashlib.sha256((OUT/name).read_bytes()).hexdigest(),'bytes':(OUT/name).stat().st_size} for name in files}}
(OUT/'independent_delivery.json').write_text(json.dumps(delivery,indent=2,sort_keys=True)+'\n')
print(json.dumps(delivery,indent=2))
raise SystemExit(not passed)
