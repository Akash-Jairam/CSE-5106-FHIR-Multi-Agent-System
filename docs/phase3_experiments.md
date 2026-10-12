# Phase 3 Experiments

## Run Orchestration (P3-19)

[`scripts/run_experiments.py`](../scripts/run_experiments.py) runs the
experiment as seven arms. Each arm is one `run_agent.py` call with its own
result file:

| Arm | System | Questions | Starting packet | Runs |
|---|---|---|---|---|
| `ordinary_direct_llm` | direct LLM | 300 held-out | full | 300 |
| `ordinary_single_agent` | single agent | 300 held-out | full | 300 |
| `ordinary_multi_agent` | multi-agent | 300 held-out | full | 300 |
| `recovery_single_agent_intact` | single agent | 50 perturbed | full | 50 |
| `recovery_single_agent_omitted` | single agent | 50 perturbed | one record left out | 50 |
| `recovery_multi_agent_intact` | multi-agent | 50 perturbed | full | 50 |
| `recovery_multi_agent_omitted` | multi-agent | 50 perturbed | one record left out | 50 |

That is 900 ordinary and 200 recovery runs. The variability arms (P3-20,
below) add 300 more. The script checks all three totals before a full
held-out run.

- **Recovery systems.** The direct baseline has no tools, so it cannot fetch
  an omitted record and has no recovery arms.
- **Intact recovery arms.** They repeat the ordinary condition on the same
  50 questions. Intact and omitted runs then come from the same batch, so
  the omitted-vs-intact comparison (P3-21) isn't confounded by when the runs
  happened.
- **Shared settings.** All arms use one `--model`, `--base_url`, and
  per-question budget (`--max_total_tokens`, `--max_tool_calls`). The tool
  cache is always on.

### The multi-agent strategy

`agent/multi_agent.py` defines `MultiAgentSystem`, the fixed controller
with the retrieval, analysis, and verification roles. It is registered as
the `multi_agent` strategy in `run_agent.py`.

### The omitted condition

With `--perturbations manifests/perturbations.json --condition omitted`,
`run_agent.py`:

1. loads the manifest with `load_perturbations()`, which checks its hashes
   against the dataset, split, packets, and code;
2. keeps only its 50 questions (in split order, before `--limit`);
3. gives each worker that question's frozen specification
   (`BaseAgent.set_perturbation()`).

The direct agent, the single agent, and the retrieval role then load the
packet with `get_perturbed_starting_packet()`. It rebuilds the original
packet from the store, checks it against the frozen hash, and removes the
one record. The store is unchanged, so a tool-equipped system can fetch the
record back. `--condition intact` selects the same 50 questions but passes
no specification. The condition and the manifest's hash are part of the
result file's run identity.

Every frozen specification matches the context the runner gives agents
(question text, patient, current time); `tests/test_phase3_orchestration.py`
checks all 50 offline.

### Resume

Resume uses the runner's existing mechanism. Results are saved every
`--save_interval` questions. Each file's `.run.json` binds it to its
inputs, settings, and code fingerprint. Completed questions are skipped,
and failed ones are retried with their earlier attempts kept in
`attempt_history`. To resume after an interruption, or to retry failures,
rerun the same `run` command.

Saved rows are checked for unique scheduled question IDs and their original
patient assignments before resume or paired analysis. Foreign, duplicate,
null-ID, or wrong-patient rows fail closed; missing rows remain unattempted.
Status distinguishes successful completion from terminal attempts. An arm
with every question attempted is analyzable even when some attempts failed:
errors must remain in the accuracy denominator, not disappear with a skipped
arm. Successful completion still controls resume, so failed rows can be
retried without losing their history.

If one arm stops (bad credentials, stale results), the remaining arms still
run, and the command exits nonzero. Any change under `agent/`, `tools/`, or
`utils/` changes the code fingerprint and makes existing files stale. In
that case, start over under a new `--output_dir`.

### Tool cache

Role retrieval replies use a bounded model-facing preview (up to 32 IDs or
Encounter records, within 8,000 serialized characters). Counts and truncation
flags distinguish that preview from the complete workspace, so large patient
searches do not flood the next model context with thousands of IDs. The
workspace and original audit reply retain all records and IDs; when a preview
differs, the audit also records its exact `model_content`.

FHIR searches are cached in `.tool_cache/` across arms and processes. The
cache now writes each entry to a private file and renames it into place,
so a parallel worker never reads a partial entry.

### Usage

From the repository root:

```sh
python scripts/run_experiments.py plan   --model gpt-5-nano --max_total_tokens 200000 --max_tool_calls 30
python scripts/run_experiments.py run    --model gpt-5-nano --max_total_tokens 200000 --max_tool_calls 30
python scripts/run_experiments.py status --model gpt-5-nano --max_total_tokens 200000 --max_tool_calls 30
```

- `plan` lists the arms, their question counts, and the exact
  `run_agent.py` commands.
- `status` reports completed, failed, and total questions per arm. It makes
  no model or FHIR calls and never prints answers.
- `--arms` selects groups (`ordinary`, `recovery`, `variability`) or single arms.
- `--split pilot --limit 2` runs a smoke test on non-held-out questions
  (ordinary arms only; select them with `--arms ordinary`). Results go to
  `output/experiments/<model>/<split>[_limit<N>]/` unless `--output_dir` is
  given, so smoke runs never mix with the real experiment.
- Every held-out `run` requires explicit positive token and
  tool ceilings. `plan` and `status` remain available without those settings.

### Limits

No live model or FHIR-store run was part of this change. The tests mock
the model and the store. The omitted condition still needs the live
store's packets to match the frozen hashes, and runtime checks fail closed
if they don't. The budget values above are examples; the experiment's
budget is chosen when it is run.

## Run-Variability Subset (P3-20)

How much of a system's accuracy is the model's run-to-run noise? Each
system answers the same fixed subset five times, and the spread across
repeats is reported. This is descriptive variability on that fixed subset;
system comparisons use the paired patient-clustered intervals (P3-21).

### The subset

`scripts/build_variability_subset.py` writes
`manifests/variability_subset.json`. It holds 20 held-out questions,
sampled with the split seed and stratified by family in proportion to the
held-out mix: 6 lookup/counting, 3 numerical aggregation, and 11 temporal
selection, from 17 patients. Only question IDs and families are read, never
reference answers. The builder checks that every held-out ID has exactly
one in-scope family assignment and that the family file matches the hash
frozen in the split. The manifest records the split manifest's hash, and
the runner refuses to use it with a different split. Two builds produced the
same file, and a test rebuilds it from the current split.

### The runs

The `variability` group has 15 arms (`variability_<system>_r1` to `_r5`),
one per system and repeat, for 3 × 5 × 20 = 300 runs. Each arm is
`run_agent.py --subset manifests/variability_subset.json --repeat K`.
`--subset` restricts the run to the subset. `--repeat` only labels the run,
but it is part of the run identity, so one repeat's file can't be resumed
as another's. All repeats share the model, budget, and tool cache. Because
FHIR results come from the same cache, the spread reflects the model and
pipeline, not the store.

### The report

```sh
python scripts/run_experiments.py variability --model gpt-5-nano --max_total_tokens 200000 --max_tool_calls 30
```

The report uses the same settings as `run` to find and check the result
files. It scores each question's first recorded attempt with the typed
scorer (an errored attempt counts as wrong). A later successful retry
cannot improve that repeat's accuracy or consistency. A current repeat
is reportable once every question has been attempted, including failures;
unattempted questions and stale files make it incomplete. Incomplete repeats
are listed as skipped. Primary tokens and latency describe those same first
attempts; retry counts and cumulative spend are reported separately. For each system it
reports:

| Measure | Meaning |
|---|---|
| accuracy, abstention rate | mean, sample SD, min and max over repeats |
| error rate | failed first attempts, separate from intentional abstentions |
| mean tokens, median latency | first-attempt usage, per repeat; unavailable if any question is unmeasured |
| retry count, retry-inclusive total tokens and latency | all recorded attempts; usage is unavailable if any attempt is unmeasured |
| always / never / sometimes correct | questions by how often they were right across repeats |
| pairwise correctness agreement | for each question, the share of repeat pairs that agree on right vs wrong, averaged |
| identical answers | share of questions answered identically in every repeat with no failed first attempt |

The report goes to `<output_dir>/variability_report.json` and the
terminal, with aggregates only. Every spread records how many repeats have
that measure available, so partial usage coverage remains visible. The statistics live in
`scripts/variability.py`, outside the fingerprinted code, so changing them
doesn't make result files stale.

### Limits

No repeats have been run yet. With 20 questions, one question is 5
accuracy points, so the SD over five repeats is a coarse estimate. It
doesn't bound the variability of future runs or replace the patient-clustered
confidence intervals (P3-21). Shared cached records hold retrieved content
fixed but do not make tools, scheduling, or provider calls deterministic.
