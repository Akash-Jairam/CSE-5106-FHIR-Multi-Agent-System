# Phase 1 Data Review

The reviewed protocol uses only lookup/counting, temporal event selection,
and numerical aggregation. `question_families.csv` retains all 2,931 source
questions, with 459 duration, paired comparison, or paired difference questions
explicitly marked `out_of_scope`. Empty and null-answer source rows remain in
the manifest and eligible rows remain available to sampling.

`development_exposure.json` records patient exposure from the proposal,
patient-specific prompt examples, the preliminary five- and twenty-question
runs, and the previous pilot. The split reserves all 32 exposed patients for
development, regardless of run correctness. The old split placed 26 patients
from prior runs or prompts in the held-out patient pool. Its results cannot be
reported as the final patient-disjoint evaluation. The regenerated split has
20 pilot, 50 dev, and 300 held-out questions from 3, 32, and 59 disjoint patient
pools. Patients in the development pool need not each have a sampled dev
question; none of their questions may enter held-out evaluation.

The original packet implementation described choosing its 8,000-token cap
using dev gold-ID recall and offered a gold-recall budget sweep. That violates
the proposal's gold-free ranking and truncation policy. This historical
exposure cannot be erased. The reviewed retriever freezes an 8,000-token input
cap as a common bounded input size, and `--sweep` now reports only token and
truncation statistics. Future changes to ranking or truncation must use only
question/record inputs on development patients. Gold-ID diagnostics run only
against an already generated packet manifest using `--gold-diagnostics`;
they must not be used to select another cap or ranking policy. Report the
previous gold-guided choice as a protocol deviation in the final report.

Packets now carry the exact rendered JSON in `resources` for tool-enabled
baselines, including only fields that the direct baseline can see. Linked
encounters, locations, specimens, and medication dependencies are included
with evidence when they fit. Typed references prevent cross-resource ID
collisions; compact and gzip NDJSON are parsed structurally. Token accounting
includes the final text, headers, and dependencies. Encounter admission
priority and organism-name code fields survive pruning so their source paths
can be cited.

The checked-in version 1 packet manifest used the superseded split and
retriever. Version 2 is regenerated from the public
[MIMIC-IV FHIR demo v2.1.0 source](https://physionet.org/files/mimic-iv-fhir-demo/2.1.0/fhir/).
All 24 source gzip files used by the packet builder were verified against the
release's [SHA256SUMS.txt](https://physionet.org/files/mimic-iv-fhir-demo/2.1.0/SHA256SUMS.txt).
The packet source manifest records their individual checksums. Regeneration:

```sh
python scripts/build_question_families.py
python scripts/build_split.py
python scripts/build_starting_packets.py --fhir-dir /path/to/fhir \
  --fhir-source-uri https://physionet.org/files/mimic-iv-fhir-demo/2.1.0/fhir/
```

Commit and freeze the resulting `manifests/starting_packets.jsonl` and
`manifests/starting_packets.source.json` before any
diagnostics or held-out experiments. All systems must use the same generated
packet version and settings. Offline regression fixtures cover compressed
and compact NDJSON parity, linked metadata, exact token ceilings, source
field/list paths, operation scope, deterministic sampling, and exposure
reservations. No held-out gold-answer analysis or live model calls were used
for this review.
