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

That is 900 ordinary and 200 recovery runs. The script checks these totals
before a full held-out run.

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
- `--arms` selects groups (`ordinary`, `recovery`) or single arms.
- `--split pilot --limit 2` runs a smoke test on non-held-out questions
  (ordinary arms only). Results go to
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
