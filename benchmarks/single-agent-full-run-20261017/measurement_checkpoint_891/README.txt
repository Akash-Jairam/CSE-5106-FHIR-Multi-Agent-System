891-question measurement-only checkpoint

IN_PROGRESS_NOT_FINAL: 891/2931 accepted measurement rows; no accuracy,
comparisons or extrapolation. All primary latency/token/proxy-cost measurements
are present. All 5,858 per-call usage records are known; actual billed cost is
unknown for all 891 accepted rows. Twenty-one runtime-failed attempts are excluded.

The original reviewed 497 script is preserved byte-for-byte. adaptation.patch and
adaptation_receipt.json show that only the output/checkpoint/proof constants and
associated descriptive labels changed. No logic, inference or provider settings
were changed. The original script was not modified.

The adapted script uses ordinary JSON parsing with strict allowlisted measurement
and identity accesses. Answer/reference/correctness fields and tool arguments are
unexamined. Selected measurement projections match the sealed raw source attempts;
the separate first-clean proof is preserved byte-identically. Canonical, ledger and
pricing hashes are equal at capture start and end. The audit writes only here.

The script refuses to overwrite an output or audit a later different checkpoint.
Keep its script/log/receipt and the measurement projection together. The original
raw attempts remain in the local run; this package contains no raw clinical data
from answers, references, traces or tool arguments. Cost estimates use the frozen
OpenAI rates as a user-authorized Azure proxy; they are not verified billing.
