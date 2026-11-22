497-question measurement-only checkpoint

IN_PROGRESS_NOT_FINAL: this records accepted per-question measurements at the
497/2931 checkpoint, without accuracy, comparisons or extrapolation. All reported
time/token/proxy-cost statistics exclude 15 runtime-failed attempts. Actual billed
cost is unknown for every accepted row.

The fixed-checkpoint script uses ordinary JSON parsing, accesses only allowlisted
measurement and identity fields, and compares them with the selected sealed raw
attempts. Answer/reference/correctness fields and tool arguments are unexamined.
Canonical, ledger and pricing SHA-256 hashes are recorded before and after capture;
any change causes refusal without inference restart. Outputs are private and no
repository or live benchmark files are written.

The script refuses to overwrite its existing output and refuses a later checkpoint
with a different accepted count. This is a provenance snapshot, not a live monitor.
Preserve the script with safe_evidence.json and per_question_measurements.jsonl.
