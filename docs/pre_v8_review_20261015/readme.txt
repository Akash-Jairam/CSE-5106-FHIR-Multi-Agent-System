Pre-V8 analysis and proposal, 15 October 2026

Start with failure_analysis.txt and proposed_design_and_tests.txt.
research_review.txt gives the research decisions and rejected assumptions;
research_sources.json pins primary URLs and newly inspected author code.

This is a bounded retrospective/research deliverable. It is not a new benchmark
run or an implemented V8. No model/FHIR requests or protected confirmation
evaluation occurred. The original goal remains paused. Existing V7 staged
completion records and all historical frozen designs are unchanged; this packet
has not been committed or pushed. No inference code was edited.

empirical/ contains reproduced 14-arm scores on the SAME exposed 300, the saved
correctness/reference matrix, 20-case purposive taxonomy (case_review.json),
compact field/tie witnesses, three strongest cases and original analysis scripts.
The original author-agent delivery manifest refers to case_taxonomy.json; it is
copied byte-identically here as case_review.json. Other copied files preserve
their names/hashes. normalization/ distinguishes architecture levers, shared
changes, actual versus possible calls, and 29 pinned authored record sources.

Full selected 20-case audit extracts include complete workspaces. They remain
PRIVATE/IGNORED under output/research/pre_v8_review_20261015/:
  selected_case_evidence.json.gz
  selected_version_specific_evidence.json.gz
Their hashes are in empirical/delivery.json. They are not bundled in these docs
or automatically published. The original ignored run outputs/audits remain the
source of truth; those extracts do not contain new/protected cases.

scientific_critique/ contains an independent critique, primary-source list and
transparent planning calculations. The calculations are illustrative independent-
pair algebra with variance-inflation sensitivity, not a patient-cluster power
estimate or a new analysis of protected results. Its review describes possible
gates; the root proposal specifies one proposed 120-case screen, still unrun.

complementary_verification/ preserves the source/author-code pins and delivery
metadata of the earlier CoVe/CRITIC/RARR audit. Raw third-party source files are
not vendored; the cited author commits are authoritative. No numerical paper
reproduction is claimed.

Reproduction guidance

The scripts are read-only with respect to project inputs/inference source, but
write analysis outputs beside themselves. Copy them to a NEW private working
directory before rerunning. They currently bind REPO to this checkout's absolute
path. Set that constant to the intended identical checkout when relocating,
recording the change. Their source/scorer/row assertions prevent silently
substituting a different experiment. Use the repository Python environment,
with FHIR_LOCAL_AZURE_CONFIG='' and FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR=''; no
credential file or provider connection is required.

cross_version.py streams only 14 existing fixed-300 result arrays and checks the
unchanged pure scorer. It reads split/family metadata for identity/stratification,
not broader QA/reference SQL. extract_cases.py and extract_version_specific.py
restore only selected 20 exposed owned audit artifacts, validating their recorded
content hashes. retained_facts.py reads those extracts and computes transparent
field/tie facts. No recorded candidate program is executed. To reproduce the
facts, regenerate/copy the two private extracts into that private working
directory rather than into versioned documentation. Exact original commands
and hashes are in empirical/delivery.json.

Research/proposal controls

The protected 1,397 IDs and existing evaluation protocol have not changed. No
new test set was sampled. The 120-case study, 60k/6-SDK/6-tool auxiliary envelope,
cache/scheduling policy and advancement gates are prospective proposals that
require implementation feasibility and a source/runtime/manifest freeze before
future execution. They do not silently amend the current inference protocol.
The final selected candidate still needs a fresh fixed-300 development comparison
and the existing once-only new-question confirmation. Separate task/model/patient
transfer remains unmeasured.
