# Shared Packet Timing Correction

Reviewing the Phase 2 retrieval scope exposed an inherited packet bug:
MedicationRequest sorting and current-time filtering used `authoredOn`
(order time), ignoring `dispenseRequest.validityPeriod.start` (prescription
start). A prescription ordered before the assumed current time could appear
in the packet even when its start was in the future. Start order could also
be reversed. Three synthetic regressions reproduce the original behavior.

The shared retriever now prefers the prescription start when present. The
packet version is incremented to 3, and all 370 frozen packets are regenerated
from the same 24 checksum-verified public MIMIC-IV FHIR demo files. The
question source, patient split, keyword ranking, 8,000-token cap, and
20/50/300 question counts are unchanged. No gold-ID diagnostics or model
calls were used to choose this correction. Previous packet versions remain
available in Git history; do not mix their outputs in one experiment.

Regenerate from local public source files:

```sh
python scripts/build_starting_packets.py --fhir-dir /path/to/fhir \
  --fhir-source-uri https://physionet.org/files/mimic-iv-fhir-demo/2.1.0/fhir/
python -m unittest discover -s tests -p 'test_phase2_packets_review.py' -v
```

All systems must use the same version-3 packet freeze. The omission harness
must also be rebuilt against it; an old perturbation manifest is not valid
after this change. Source downloads, runtime settings, model launchers, and
credentials stay local. Only the shared implementation, regression tests,
and reproducibility manifest belong in the correction PR.
