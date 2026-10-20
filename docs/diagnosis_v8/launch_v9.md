# V9 launch checklist (credentialed machine only)

Branch `experiments/dual-solver-v9` (built on `experiments/diagnosis-v8`, i.e.
the frozen V8 code `dd0f7d8` plus the V8 records). Nothing below has been run;
the diagnosis machine has no credentials or FHIR store. Same conventions as
[`launch.md`](launch.md): every command is in `run_experiments.py` form, takes
the same flags through the per-user launcher, always uses `--split held_out`,
and runs under `caffeinate -i` on AC power.

The frozen design and gate are in [`design_v9.txt`](design_v9.txt). Its
pre-registered replay predicts **246 / 300, the same as the single agent**
(bound 250 with a perfect adjudicator). Launching anyway is a deliberate
choice: the run is the prespecified test of the plan.

## 0. Pre-flight (offline, ~1 min)

```
git checkout experiments/dual-solver-v9
.venv/bin/python -m pytest -q tests        # expect 640 passed, 2 skipped
python scripts/check_setup.py --model gpt-6-luna
python scripts/run_experiments.py plan --split held_out --arms ordinary_single_agent ordinary_multi_agent \
    --team_strategy multi_agent_v9 --model gpt-6-luna --max_total_tokens 200000 --max_tool_calls 30 \
    --num_processes 4 --save_interval 1 --audit_storage gzip --output_dir output/experiments/gpt-6-luna/v9_300q
```

The `plan` output must show `--agent_strategy multi_agent_v9` for
`ordinary_multi_agent` and `single_agent` for the control (checked on the
diagnosis machine).

Five-question smoke (not an experiment entry; throwaway dir):

```
caffeinate -i python scripts/run_experiments.py run --split pilot --limit 5 \
    --arms ordinary_single_agent ordinary_multi_agent --team_strategy multi_agent_v9 \
    --model gpt-6-luna --max_total_tokens 200000 --max_tool_calls 30 \
    --audit_storage gzip --num_processes 2 --save_interval 1 --output_dir output/smoke/v9_pilot5
python scripts/replay_v9_controller.py mechanism --team output/smoke/v9_pilot5/ordinary_multi_agent.json
```

Check that every team row has `revision.protocol == "dual-solver-v9"` and a
`release_policy` in the seven listed in `design_v9.txt`. `unknown_policies`
must be `[]`, and no terminal error may come from `agent/dual_solver_agent.py`.
Also check that `phases` lists `primary` then `recompute` (unless the policy
is `recompute_skipped_budget`), and that `tokens.total` per row ≤ 200,000.

## 1. Freeze (before the first held_out attempt)

```
DATE=$(date +%Y%m%d)
python scripts/record_design.py freeze --record benchmarks/design-v9-$DATE \
    --description docs/diagnosis_v8/design_v9.txt \
    --output_directory output/experiments/gpt-6-luna/v9_300q \
    --model gpt-6-luna --max_total_tokens 200000 --max_tool_calls 30 --questions 300 \
    --arm ordinary_single_agent --arm ordinary_multi_agent \
    --command_text "see docs/diagnosis_v8/launch_v9.md section 2"
```

Commit `benchmarks/design-v9-$DATE/{design.txt,manifest.json}` (with
source.zip, as earlier records did). Do not edit `design_v9.txt` after this
point.

## 2. Paired 300 under the V9 fingerprint (600 first attempts)

`agent/` changed, so the single control runs fresh. The direct LLM is
**not** re-run (its code path is unchanged; see section 3).

```
caffeinate -i python scripts/run_experiments.py run --split held_out \
    --arms ordinary_single_agent ordinary_multi_agent --team_strategy multi_agent_v9 \
    --model gpt-6-luna --max_total_tokens 200000 --max_tool_calls 30 --num_processes 4 \
    --save_interval 1 --audit_storage gzip --output_dir output/experiments/gpt-6-luna/v9_300q
```

First-attempt discipline: do **not** re-issue `run` into the same
`--output_dir` after a crash or provider timeout. Terminal errors stay in the
denominator as wrong.

## 3. Analysis (offline, no credentials)

```
D=output/experiments/gpt-6-luna/v9_300q
C=output/experiments/gpt-6-luna/v8_300q          # run C, for the reused direct arm
python scripts/run_experiments.py analyze --split held_out --arms ordinary_single_agent ordinary_multi_agent \
    --team_strategy multi_agent_v9 --audit_storage gzip --output_dir $D \
    --max_total_tokens 200000 --max_tool_calls 30 --model gpt-6-luna
python scripts/run_experiments.py revisions --split held_out --arms ordinary_single_agent ordinary_multi_agent \
    --team_strategy multi_agent_v9 --audit_storage gzip --output_dir $D \
    --max_total_tokens 200000 --max_tool_calls 30 --model gpt-6-luna

# gate items 1-2, plus the reused direct arm (compare_runs accepts differing fingerprints)
python scripts/compare_runs.py --split held_out \
    --arm single=$D/ordinary_single_agent.json \
    --arm team=$D/ordinary_multi_agent.json \
    --arm direct_runC=$C/ordinary_direct_llm.json \
    --output benchmarks/design-v9-$DATE/results.json

# gate item 3 and the mechanism table (policies, transitions per policy, adjudication, phase tokens)
python scripts/replay_v9_controller.py mechanism --team $D/ordinary_multi_agent.json \
    --output benchmarks/design-v9-$DATE/mechanism.json

python scripts/record_design.py finish --record benchmarks/design-v9-$DATE \
    --report benchmarks/design-v9-$DATE/results.json
```

`analyze` and `revisions` need `--team_strategy multi_agent_v9 --audit_storage
gzip`, or both arms are rejected as stale (V8 hand-off note).

Gate (frozen): team > single in `results.json` → `arms.*.accuracy.all.correct`;
the patient-block CI lower bound of `team - single` > −2 pts; and in
`mechanism.json`, `replacements.adjudicated_recompute.right_to_wrong` ≤
`wrong_to_right`. A released `adjudicated_primary` can never change an answer.
Write `benchmarks/design-v9-$DATE/outcome.txt` either way. Also record live
recompute availability: rows not `recompute_skipped_budget`, against the
pre-run estimate of 236/300.

Send back what went back for V8: the three arm files (`audit/` not needed),
`analysis/`, `revisions/`, `results.json`, `mechanism.json`, `outcome.txt`.

## 4. Confirmation (only if the gate passes)

Protected questions are every id not in `split.json` pilot/dev/held_out, on a
patient not in `manifests/development_exposure.json` or
`docs/diagnosis_v8/audited_subject_ids.json`. Draw a family-stratified 300
(seed 5106) once and write it as a split manifest. Run single + team once with
the section-2 settings and score once with `compare_runs.py`. No reruns and no
second sample.
