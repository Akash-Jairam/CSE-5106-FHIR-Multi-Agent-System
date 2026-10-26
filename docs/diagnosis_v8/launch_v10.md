# V10 launch checklist (credentialed machine only)

Branch `experiments/finisher-team-v10`, built on `experiments/dual-solver-v9`
at `bb13a58` (V9 code plus the B2 record). The design, prediction and gate are
in [`design_v10.txt`](design_v10.txt). Same conventions as
[`launch_v9.md`](launch_v9.md): every command is in `run_experiments.py`
form, uses `--split held_out` for the experiment, and runs under
`caffeinate -i` on AC power. Model `gpt-5.4-nano` (as in B2), 200,000 tokens
and 30 tools per question for every arm.

Nothing below §1 has been run.

## 0. Pre-flight (offline, ~1 min)

```
git checkout experiments/finisher-team-v10
.venv/bin/python -m pytest -q tests        # expect 679 passed, 2 skipped
python scripts/check_setup.py --model gpt-5.4-nano
python scripts/run_experiments.py plan --split held_out --arms ordinary_single_agent ordinary_multi_agent \
    --team_strategy multi_agent_v10 --model gpt-5.4-nano --max_total_tokens 200000 --max_tool_calls 30 \
    --num_processes 4 --save_interval 1 --audit_storage gzip --output_dir output/experiments/gpt-5.4-nano/v10_300q_r1
```

The `plan` output must show `--agent_strategy multi_agent_v10` for
`ordinary_multi_agent` and `single_agent` for the control.

Five-question smoke (done on the diagnosis machine; see `design_v10.txt`,
"Smoke"; repeat only if the code changes):

```
caffeinate -i python scripts/run_experiments.py run --split pilot --limit 5 \
    --arms ordinary_single_agent ordinary_multi_agent --team_strategy multi_agent_v10 \
    --model gpt-5.4-nano --max_total_tokens 200000 --max_tool_calls 30 \
    --audit_storage gzip --num_processes 2 --save_interval 1 --output_dir output/smoke/v10_pilot5
python scripts/replay_v10.py mechanism --team output/smoke/v10_pilot5/ordinary_multi_agent.json
```

Every team row must have `revision.protocol == "finisher-team-v10"`, a `vote`
in `agree2|majority3|split|single_draw`, a `release_policy` in the five listed
in `design_v10.txt`, and a `finisher.finisher` in
`none|skipped_no_anchor|confirmed_primary|replaced`. `unknown_policies`,
`unknown_votes` and `unknown_finisher_outcomes` must be `[]`. `phases` lists
`draw1`, then `draw2` unless `budget_plan.draw2.skip` is set. Every row has
`tokens.total` ≤ 200,000 and `budget.tool_calls_run` ≤ 30.

## 1. Freeze (before the first held_out attempt)

```
DATE=$(date +%Y%m%d)
python scripts/record_design.py freeze --record benchmarks/design-v10-$DATE \
    --description docs/diagnosis_v8/design_v10.txt \
    --output_directory output/experiments/gpt-5.4-nano/v10_300q_r1 \
    --model gpt-5.4-nano --max_total_tokens 200000 --max_tool_calls 30 --questions 300 \
    --arm ordinary_single_agent --arm ordinary_multi_agent \
    --command_text "see docs/diagnosis_v8/launch_v10.md section 2 (two repeats, r1 and r2)"
```

Commit `benchmarks/design-v10-$DATE/{design.txt,manifest.json}` (with
`source.zip`). Do not edit `design_v10.txt` after this point.

## 2. Two repeats of the paired 300 (1,200 first attempts)

`agent/` and `tools/` changed (V10 controller and the shared
`get_linked_resources` tool), so the single runs fresh under the V10
fingerprint. Each repeat is its own output directory; the second starts
only after the first has finished. Both repeats belong to the same frozen
record.

```
for R in r1 r2; do
  caffeinate -i python scripts/run_experiments.py run --split held_out \
      --arms ordinary_single_agent ordinary_multi_agent --team_strategy multi_agent_v10 \
      --model gpt-5.4-nano --max_total_tokens 200000 --max_tool_calls 30 --num_processes 4 \
      --save_interval 1 --audit_storage gzip --output_dir output/experiments/gpt-5.4-nano/v10_300q_$R
done
```

First-attempt discipline: do **not** re-issue `run` into the same
`--output_dir` after a crash or provider timeout. Terminal errors stay in the
denominator as wrong. The B2 billing rule applies unchanged: if attempts fail
with a provider billing error, keep the raw file, retry only those attempts
into a derived `*.billing_retried.json` after a top-up, and report both scores.

## 3. Analysis (offline, no credentials)

```
B=benchmarks/design-v10-$DATE
for R in r1 r2; do
  D=output/experiments/gpt-5.4-nano/v10_300q_$R
  for C in analyze revisions; do
    python scripts/run_experiments.py $C --split held_out --arms ordinary_single_agent ordinary_multi_agent \
        --team_strategy multi_agent_v10 --audit_storage gzip --output_dir $D \
        --max_total_tokens 200000 --max_tool_calls 30 --model gpt-5.4-nano
  done
  python scripts/compare_runs.py --split held_out \
      --arm single=$D/ordinary_single_agent.json --arm team=$D/ordinary_multi_agent.json \
      --output $B/results_$R.json
  python scripts/replay_v10.py mechanism --team $D/ordinary_multi_agent.json --output $B/mechanism_$R.json
done
# all four arms in one paired table (both repeats, cross-repeat single-single = draw noise)
python scripts/compare_runs.py --split held_out \
    --arm single_r1=output/experiments/gpt-5.4-nano/v10_300q_r1/ordinary_single_agent.json \
    --arm team_r1=output/experiments/gpt-5.4-nano/v10_300q_r1/ordinary_multi_agent.json \
    --arm single_r2=output/experiments/gpt-5.4-nano/v10_300q_r2/ordinary_single_agent.json \
    --arm team_r2=output/experiments/gpt-5.4-nano/v10_300q_r2/ordinary_multi_agent.json \
    --output $B/results.json
python scripts/record_design.py finish --record $B --report $B/results.json
```

Gate (frozen in `design_v10.txt`), evaluated on the mean over the two repeats
and reported per repeat:

1. team correct > fresh single correct (`results_r*.json` →
   `arms.*.accuracy.all.correct`);
2. patient-block 95 % CI lower bound of `team − single` > −2 points in each
   repeat's `results_r*.json`;
3. finisher harm: in each `mechanism_r*.json`,
   `finisher_replacements.total.right_to_wrong` ≤ `wrong_to_right`
   (`gate_finisher_harm_ok`);
4. noise check on the mean: the mean of `mechanism_r1.json` and
   `mechanism_r2.json` → `draw1_correct` within ±6 of the mean fresh single
   correct over `results_r1.json` and `results_r2.json`. If outside, the
   comparison is noise-dominated: report it, do not interpret it. Report each
   repeat's draw-1 − single difference as well (not a gate item).

Write `$B/outcome.txt` either way, with the per-repeat paired differences, the
mean, and the single_r1 − single_r2 difference (the measured draw noise).

Send back: the four arm files (`audit/` not needed), `analysis/`, `revisions/`,
`results*.json`, `mechanism_r*.json`, `outcome.txt`.

## 4. Confirmation (only if the gate passes)

Single + team once on the protected `manifests/confirmation_v9.json` sample,
with the §2 settings, scored once. That sample was drawn once for V9 and used
once, for the V9 confirmation on gpt-6-luna; this is its second use and its
first with nano, as `design_b2.txt` already allowed. No reruns and no second
sample.

```
D=output/experiments/gpt-5.4-nano/v10_confirmation
caffeinate -i python scripts/run_experiments.py run --split held_out --split_manifest manifests/confirmation_v9.json \
    --arms ordinary_single_agent ordinary_multi_agent --team_strategy multi_agent_v10 \
    --model gpt-5.4-nano --max_total_tokens 200000 --max_tool_calls 30 --num_processes 4 \
    --save_interval 1 --audit_storage gzip --output_dir $D
python scripts/compare_runs.py --split_manifest manifests/confirmation_v9.json --split held_out \
    --arm single=$D/ordinary_single_agent.json --arm team=$D/ordinary_multi_agent.json \
    --output $B/confirmation_results.json
python scripts/replay_v10.py mechanism --team $D/ordinary_multi_agent.json --output $B/confirmation_mechanism.json
```
