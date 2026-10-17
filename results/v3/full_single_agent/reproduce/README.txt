Offline analysis source preservation

sourceanalysis/package.py and base.py are the exact executed consolidation
analysis sources. Their hashes are in analysis_source_manifest.json. They are
analysis/archive code, not inference implementations. No credentials or provider
configuration values are included.

The preserved extractor records its historical workspace/run locations and
hash-bound full dataset, frozen source files and original sealed attempts. It
used the unchanged repository typed scorer, question_record/summarize/paired
functions and experiment_stats bootstrap. Those project implementation files
are identified by hashes in internal/, not duplicated here. This historical
extractor is evidence of how these numeric results were obtained; do not execute
it into the existing result directory because it writes analysis outputs.

For immediate portable data recovery use ../raw/RECONSTRUCTION.txt. All parsed
accepted result fields and exact owned audit UTF8 payload bytes can be restored
from the data archive using the standard-library archive tool. There are no
provider/model calls in that operation. The saved per-question flags/numerical
records and cohort IDs also permit offline arithmetic and paired reanalysis
without touching provider setup.

Fresh reproductions should use NEW output locations, preserve source IDs and
patient-cluster assignments, keep cached tokens a subset of prompt tokens, and
retain the difference convention Single Agent minus V3. The historical frozen
patient bootstrap uses10000 resamples, seed5106 and question-weighted means.
Original full2931 completion gates remain unchanged. This analysis separately
uses the explicitly authorized matched completed cohort N2929.

The shared portable data packager is results/v3/full_v3/pack_archive.py; the shared verify/reconstruct reader is results/v3/full_v3/archive_tools.py. Their source hashes are recorded in raw/archive and portable verification metadata.
