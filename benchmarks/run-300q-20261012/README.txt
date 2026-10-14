Run snapshot: run-300q-20261012

This snapshot preserves the implementation and results of the first completed
experiment on the proposal's 300-question subset. Use the annotated Git tag
run-300q-20261012 when citing this implementation. The companion branch is
runs/300q-20261012. The tag must remain at the original snapshot commit; use a
new branch, run directory, and tag for subsequent experiments.

What the 300 questions mean

The original FHIR-AgentBench authors evaluated all 2,931 questions. Our proposal
explicitly specifies a subset study: 300 held-out questions spanning lookup /
counting, temporal selection, and numerical aggregation. It does not claim an
official full-benchmark result. The original paper is available at
https://arxiv.org/html/2509.19319v2 and the preserved proposal is proposal.pdf.

scripts/build_split.py uses seed 5106, reserves proposal examples and documented
previously exposed patients for development, assigns other patients to disjoint
pilot and held-out pools, and samples questions in proportion to the three
in-scope families' dataset frequencies. The exact IDs are in manifests/split.json.
The 300 selected questions cover 58 patients: 92 lookup/counting, 159 temporal
selection, and 49 numerical aggregation. The allocation pool contains 59 held-out
patients; one has no question selected in this 300-question sample.

Calling this a completed run means completion of the proposed experiment, not
evaluation of all 2,931 questions. Sampling excludes other question families and
does not guarantee the full benchmark's difficulty or resource-type distribution.

Architectures and run settings

All systems used GPT-6 Luna through the configured Azure deployment, with
reasoning_effort none, identical question context and starting packets, a shared
200,000 aggregate-token / 30-tool-call ceiling, and an inherited estimated-input
guard of 32,000 tokens. The local adapter capped completions at 16,384 tokens.

Direct LLM: one response using the starting packet, without tools or revision.
Single agent: modified Multi-turn + Retriever + Code, one conversation context,
starting packet, Python execution, deterministic field checks, and one revision
opportunity following a failed final evidence check.
Verified team: separate retrieval, analysis, and record-verification contexts,
a fixed controller, structured evidence, and at most one targeted repair.

The source is parent commit d87ae5a2e7d7e57fceeb17b88e5b9d877b15361c plus the
five-file pilot patch recorded in results/pilot_fix.patch. That patch clarifies
the Python return contract and permits a bounded final role response after the
last tool result. No post-run repair was applied to the evaluated implementation.
source_sha256.json records the evaluated contents of all previously tracked files.
runtime/ preserves the non-credential local adapter and dependency freeze.
The original provider binding and launch manifest are copied under results/.

Results preserved

900 ordinary attempts: 300 questions x direct / single / team.
200 recovery attempts: 50 questions x intact / omitted x single / team.
300 variability attempts: 20 questions x 5 repetitions x 3 systems.
All 1,400 first attempts are retained, including 102 execution errors.
No retries were used to replace first attempts in the primary outcomes.

Ordinary typed-answer accuracy:
  Direct LLM:   202/300 = 67.3%
  Single agent: 223/300 = 74.3%
  Verified team:162/300 = 54.0%
Team minus single: -20.3 percentage points; patient-clustered 95% bootstrap
interval [-26.4, -14.1]. These are results for this implementation and subset,
not directly comparable with the original paper's full-set LLM-judge scores.

results/ contains byte-identical copies of 55 compact reports, figures, run
identities, and operational records. investigations/ contains 22 subsequent
diagnostic, provenance, and cost files. Subsequent investigation and the repair
plan are observational/post-run material, not additional inference or fixes.
The current repair plan is preserved at docs/midterm_multi_agent_repair_plan.txt.

The cost estimate is approximately $2.69 using recorded usage and Azure Global
Standard cached-read list-price assumptions. Deployment/contract charges are not
verified; one timeout has unknown usage. See the preserved cost_summary.txt for
other rate scenarios and exclusions. Saved null cost fields mean unavailable.

Known limitations to disclose in reports

The pre-run Luna smoke pilot used three questions twice, rather than completing
the promised 20-question pilot and roughly 50-question development campaign.
The blinded two-human 30-50-output audit and official secondary LLM judge were
not completed. Citation validity is not semantic support or completeness.
Successful omitted-resource use is incompletely recorded for the single agent.
The frozen primary scorer can credit empty/null answers without independently
established empty-result evidence. Keep that primary scoring unchanged and label
additional support-aware analyses separately.
Earlier packet-cap selection used development gold-ID recall; this departure
from the proposal's gold-free packet-selection rule is recorded in
docs/phase1_data_review.md. It is not evidence that agents received held-out gold
answers. Repairs informed by this run make future same-set results exploratory;
use fresh unexposed patients for confirmatory improvement claims.

Raw-result archive and recovery

All 101 original run files are retained in
output/archives/run-300q-20261012-results.tar.gz (about 445 MB compressed,
1,171,375,127 original bytes). Its SHA-256 and original per-file inventory are
recorded in snapshot_manifest.json and baseline_inventory.json.
The original output/experiments/gpt-6-luna/midterm_20261012_full_v1 directory
is unchanged. The raw archive is kept outside Git; reports and hashes are in Git.
This branch/tag and archive are local pending a separate GitHub backup.

From the repository root, verify without inference:
  python3 benchmarks/run-300q-20261012/verify_snapshot.py

To check a downloaded or relocated archive:
  python3 benchmarks/run-300q-20261012/verify_snapshot.py --archive PATH

The verifier checks the tag's source files, copied artifacts, archive checksum,
and every archived original file. A mismatch must be investigated without
rewriting the original results or moving the tag.

For durable remote preservation, push the branch and annotated tag and attach
the raw archive plus checksum to a GitHub Release or an archival storage service.
Credential-bearing configuration and caches must remain excluded. Credentials
must be supplied locally to reproduce inference; they are not part of this
snapshot. All future paid inference must use a new output directory.
