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

If one arm stops (bad credentials, stale results), the remaining arms still
run, and the command exits nonzero. Any change under `agent/`, `tools/`, or
`utils/` changes the code fingerprint and makes existing files stale. In
that case, start over under a new `--output_dir`.

### Tool cache

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

### Limits

No live model or FHIR-store run was part of this change. The tests mock
the model and the store. The omitted condition still needs the live
store's packets to match the frozen hashes, and runtime checks fail closed
if they don't. The budget values above are examples; the experiment's
budget is chosen when it is run.

## Run-Variability Subset (P3-20)

How much of a system's accuracy is the model's run-to-run noise? Each
system answers the same fixed subset five times, and the spread across
repeats is reported. Differences between systems smaller than that spread
should not be read as real.

### The subset

`scripts/build_variability_subset.py` writes
`manifests/variability_subset.json`. It holds 20 held-out questions,
sampled with the split seed and stratified by family in proportion to the
held-out mix: 6 lookup/counting, 3 numerical aggregation, and 11 temporal
selection, from 17 patients. Only question IDs and families are read, never
reference answers. The manifest records the split manifest's hash, and the
runner refuses to use it with a different split. Two builds produced the
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
files. It scores each answer with the typed scorer (an errored attempt
counts as wrong), and only uses repeats whose files are complete and
current. Incomplete repeats are listed as skipped. For each system it
reports:

| Measure | Meaning |
|---|---|
| accuracy, abstention rate | mean, sample SD, min and max over repeats |
| mean tokens, median latency | the same, per repeat |
| always / never / sometimes correct | questions by how often they were right across repeats |
| pairwise correctness agreement | for each question, the share of repeat pairs that agree on right vs wrong, averaged |
| identical answers | share of questions answered identically in every repeat |

The report goes to `<output_dir>/variability_report.json` and the
terminal, with aggregates only. The statistics live in
`scripts/variability.py`, outside the fingerprinted code, so changing them
doesn't make result files stale.

### Limits

No repeats have been run yet. With 20 questions, one question is 5
accuracy points, so the SD over five repeats is a coarse estimate. It
bounds run-to-run noise but doesn't replace the patient-clustered
confidence intervals (P3-21). Models called at temperature 0 may still
vary through provider nondeterminism; reasoning models ignore the setting.

## Statistics and Plots (P3-21)

```sh
python scripts/run_experiments.py analyze --model gpt-5-nano --max_total_tokens 200000 --max_tool_calls 30
```

`analyze` uses the same settings as `run` to find and check the ordinary
and recovery result files. It analyzes only arms whose files are complete
and current, and lists the rest as skipped. Answers are scored with the
typed scorer (an errored attempt counts as wrong), and no answer is
written or printed.

### Intervals

Questions from the same patient share one record, so they aren't
independent. Every interval is a 95% percentile bootstrap that resamples
**patients** with replacement and keeps all of a resampled patient's
questions (`scripts/experiment_stats.py`). The held-out set has 300
questions from 58 patients; the perturbed set has 50 from 27. Resampling
uses 10,000 draws (`--bootstrap`) seeded with the split seed, so rerunning
the analysis gives identical intervals.

**Paired differences** compare two arms on the same questions. Each
resample takes the same patients for both arms, so the difference doesn't
include between-question variation. Each difference also has a two-sided
bootstrap p-value: twice the share of resamples on the far side of zero.

| Comparison | Arms (a − b) |
|---|---|
| systems | multi − single, multi − direct, single − direct (ordinary arms, 300 questions) |
| intact vs omitted | omitted − intact for the single agent and for the multi-agent system (50 questions) |
| recovery between systems | multi − single, under omitted and under intact |

Each comparison is reported overall and for each question family.

### Outputs

The command writes these files to `<output_dir>/analysis/`:

| File | Contents |
|---|---|
| `accuracy.csv` | accuracy per arm, overall and by family, with n, patients, and the 95% CI |
| `comparisons.csv` | every paired difference, overall and by family, with its CI and p-value |
| `latency.csv` | median and p95 latency, and mean tokens per question with a clustered CI, per arm |
| `accuracy_vs_tokens.svg` | the three ordinary arms: accuracy against mean tokens per question |
| `summary.json` | all of the above, plus the settings and the skipped arms |

The chart has a log token axis and 95% CI whiskers on both axes. Each
system has a fixed color from a palette checked for color-vision
deficiency in light and dark mode, plus a direct label and a hover tooltip.
It is a self-contained SVG with no plotting dependency, and the CSVs carry
the same numbers as a table. The analysis code lives in `scripts/`, outside
the fingerprinted code, so changing it doesn't make result files stale.

### Limits

No real results have been analyzed yet. The tests use synthetic result
files. Family-level and perturbed-set intervals are wide: the smallest
family has 49 held-out questions, and the perturbed set has 50 questions
from 27 patients. The p-values are not adjusted for multiple comparisons.

## Revision Tracking (P3-22)

```sh
python scripts/run_experiments.py revisions --model gpt-5-nano --max_total_tokens 200000 --max_tool_calls 30 \
  [--evidence_source store | --evidence_source ndjson --fhir_dir "$FHIR_NDJSON_DIR"]
```

`revisions` reads the complete, current ordinary and recovery arms. It
writes `revisions.json` (a summary per arm) and `revision_log.csv` (one row
of flags per question and arm) to `<output_dir>/revisions/`. Answers are
scored with the typed scorer but never written or printed. The log holds
question IDs and true/false flags, so it can be used to sample questions
for the human audit (P4-24).

| Measure | Definition |
|---|---|
| draft → released | The single agent and the multi-agent system save their first draft in `revision`. Scoring the draft and the released answer gives right→right, right→wrong, wrong→right, or wrong→wrong. The released answer is the one evaluated, so a correct draft withheld as unverified counts as right→wrong. Counts are given over all runs with a draft and over repaired runs only, each with a net (wrong→right − right→wrong). The log also records the repair role. |
| unsupported claims | A released, non-abstaining answer with no evidence. With `--evidence_source`, it also counts an answer with any citation that doesn't match its source record (`utils/evidence_scorer.py`). Without an evidence source, only the no-evidence count is a real count; it is a lower bound on unsupported claims. |
| empty-answer correctness | On questions whose reference is empty (`[]` or `[[None]]`), the share answered correctly by abstaining or saying none/zero. **False abstentions** are abstentions on questions that do have a reference answer. |
| omitted-resource recovery | On the omitted packet: whether the removed record is among the run's retrieved resources, whether it is cited, and whether the answer is right. It also reports accuracy when the record was fetched back, and accuracy on questions the same system got right from the intact packet. |

The direct baseline has no revisions, so it reports no transitions. The
multi-agent system's retrieved resources are the workspace shared by
retrieval and analysis. Records only the verifier fetched don't count as
recovered, because they never reached the answer.

### Limits

No real runs have been tracked yet; the tests use synthetic result files.
On the 50 perturbed questions, a recovery rate of 1 question is 2
percentage points. Use the `analyze` intervals (P3-21) before comparing
systems.

## Demo Interface (P3-23)

```sh
bash scripts/run_demo.sh --model gpt-5-nano --max_total_tokens 200000 --max_tool_calls 30
```

The launcher changes to the repository root and runs `streamlit run demo/app.py` (Streamlit is already in
`requirements.txt`). It binds to `localhost` and hides Streamlit's deploy
toolbar.

- **Patient selection.** The sidebar picks the system (multi-agent by
  default, single agent, or direct LLM) and a patient. Patients come from
  the pilot and dev splits; `--splits` changes this. Held-out patients
  aren't offered by default, so the demo doesn't add exposure to the
  evaluation set.
- **Example or custom question.** An example is one of the patient's
  dataset questions, with its assumed current time. A custom question takes
  free text and an optional current time (`YYYY-MM-DD HH:MM:SS`).
  `demo/backend.build_input()` builds the same context string the runner
  gives agents, which a test checks against a dataset row. The picker reads
  only question, patient, and time-context columns, after validating the
  dataset's hash against the split. Reference answers are never passed to
  the interface. Changing the system, patient, question, or assumed time
  clears the previous result; an unsuccessful new request clears it too.
  Custom input is parsed back through the runner's context parser; conflicting
  embedded patient IDs, timestamps, or `Context:` sections are rejected so
  the selected patient and assumed time remain authoritative.
- **Answer, cited fields, verification status.** The result shows:
  - the released answer (or "No answer") and any limitation;
  - a table of cited fields (resource, field, value), marked `ok` only with
    a recorded clean check, `not confirmed` when check results are absent or
    incomplete, or with the remaining issue on that field or record;
  - a status badge with an icon and label:

    | Status | Meaning |
    |---|---|
    | ✓ Verified | a completed check was recorded, clean, and the answer was released |
    | ✓ Verified empty result | the multi-agent final explicitly returned null and the independent check accepted it; the answer displays "No matching value" |
    | ⚠ Not verified | check issues remained; the badge states whether an answer was released or withheld |
    | ○ Not checked | the direct baseline, or a released answer with no recorded completed check |
    | – No answer | the system abstained or hit its budget |
    | ✕ Error | the run failed |

  The badge also notes whether a repair ran, and where it started. Expanders
  show the draft and first-check issues, and the usage (tokens, model calls,
  tool calls, latency, cost). Missing usage is shown as unavailable.
- **Backed by the evaluated pipeline.** `backend.ask()` calls the runner's
  `run_agent_with_input()` with the same strategy, model, base URL,
  per-question budget, and tool cache as the experiment. The header compares
  the current code fingerprint, model, base URL, and budget with the
  `.run.json` of the evaluated ordinary arm for that system (from
  `--output_dir`, default as in `run_experiments.py`). A matching badge also
  requires the runner's exact full identity, including the input hash,
  a current split/source binding, and a result file with every
  planned held-out question attempted exactly once per result row, with
  matching question, patient, and time context. Metadata alone or a partial
  smoke run cannot claim an evaluated batch. The header reports incomplete
  batches or identity differences. Offline identity validation reads hidden
  reference labels through `run_agent.load_inputs()` and `run_identity()`;
  these labels are used only for the hash and are never passed to the model
  or interface. The patient picker continues to read context columns only.

The tests check the backend functions and run the app headless with
Streamlit's `AppTest`, with the model and store mocked. During
development, I also rendered and inspected the verified and not-verified
states in a browser. A live question needs the model API key and the FHIR
store configured; otherwise the app says so and doesn't run.
