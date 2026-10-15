# V8 / B1 / B2 launch checklist (credentialed machine only)

Nothing below has been run. This machine has no Azure credentials or FHIR
store, so the runs are prepared here and launched by the team. Prior frozen
runs were launched through the per-user launcher
(`.local-setup/phase3.py run --project-root . ...`), which wraps
`scripts/run_experiments.py run` with the provider bootstrap; every command
below is given in `run_experiments.py` form and takes the same flags.

Naming: the "300 development questions" used by V2–V7 and in
`decision.md` are `manifests/split.json` → `held_out` (300 ids, 59 patients).
Always pass `--split held_out`; `--split dev` is the 50-question smoke split.

## 0. Pre-flight (offline, ~1 min)

```
git checkout experiments/diagnosis-v8
.venv/bin/python -m pytest -q tests                       # expect all passed
python scripts/check_setup.py --model gpt-6-luna
```

Five-question smoke of the new strategy on the pilot split (not an
experiment entry; output dir is throwaway):

```
python scripts/run_experiments.py run --split pilot --limit 5 --arms ordinary_multi_agent \
    --team_strategy multi_agent_v8 --audit_storage gzip --num_processes 2 --save_interval 1 \
    --output_dir output/smoke/v8_pilot5
```

Check the result rows carry `revision.protocol == "convention-verified-v8"`
and a `release_policy`, and that no row is a terminal error caused by the
checks themselves (`agent/convention_checks.py` must never raise).

## 1. Freeze (before the first held_out attempt)

```
DATE=$(date +%Y%m%d)
python scripts/record_design.py freeze --record benchmarks/design-v8-$DATE \
    --description docs/diagnosis_v8/design_v8.txt \
    --output_directory output/experiments/gpt-6-luna/v8_300q \
    --model gpt-6-luna --max_total_tokens 200000 --max_tool_calls 30 --questions 300 \
    --arm ordinary_direct_llm --arm ordinary_single_agent --arm ordinary_multi_agent \
    --command_text "see docs/diagnosis_v8/launch.md section 2"
```

Commit `benchmarks/design-v8-$DATE/{design.txt,manifest.json}` (source.zip
as the earlier records did). Do not edit `design_v8.txt` after this point.

## 2. C — paired 300 under the new fingerprint (900 first attempts)

B3 changed `agent/ehr_semantics.py` and `agent/direct_llm_agent.py`, so no
earlier control is reusable; all three arms run fresh.

```
python scripts/run_experiments.py run --split held_out --arms ordinary \
    --team_strategy multi_agent_v8 --model gpt-6-luna \
    --max_total_tokens 200000 --max_tool_calls 30 --num_processes 4 --save_interval 1 \
    --audit_storage gzip --output_dir output/experiments/gpt-6-luna/v8_300q
```

First-attempt discipline: do **not** re-issue `run` into the same
`--output_dir` after a worker crash or provider timeout (resume retries failed
rows and would violate the frozen protocol). Terminal errors stay in the
denominator as incorrect.

## 3. D2 arm 2 — heterogeneous verifier (300 team attempts)

Separate output dir; only the static program audit moves to the other model.
Pick a different model family that passes `check_setup.py`.

```
python scripts/run_experiments.py run --split held_out --arms ordinary_multi_agent \
    --team_strategy multi_agent_v8 --verifier_model <other-family-deployment> \
    --model gpt-6-luna --max_total_tokens 200000 --max_tool_calls 30 --num_processes 4 \
    --save_interval 1 --audit_storage gzip \
    --output_dir output/experiments/gpt-6-luna/v8_300q_hetero
```

Compared against the single control from section 2 (same fingerprint; the
verifier model is in the run identity).

## 4. B1 — recovery rerun (200 first attempts)

Recovery arms require `--split held_out` (the 50 perturbed questions are a
frozen subset of it).

```
python scripts/run_experiments.py run --split held_out --arms recovery \
    --team_strategy multi_agent_v8 --model gpt-6-luna \
    --max_total_tokens 200000 --max_tool_calls 30 --num_processes 4 --save_interval 1 \
    --audit_storage gzip --output_dir output/experiments/gpt-6-luna/v8_300q
```

(Same output dir as section 2 is fine: arm files are
`recovery_{single_agent,multi_agent}_{intact,omitted}.json`, and `analyze`
reads ordinary + recovery together.)

## 5. B2 — weaker base model (900 first attempts)

Fix the model before launch: smallest deployment passing
`python scripts/check_setup.py --model <m>` and a 5-question pilot smoke
(candidate order `gpt-5-mini`, `gpt-5.4-mini`). Record the choice in
`report.md`.

```
python scripts/run_experiments.py run --split held_out --arms ordinary \
    --team_strategy multi_agent_v8 --model <weaker> \
    --max_total_tokens 200000 --max_tool_calls 30 --num_processes 4 --save_interval 1 \
    --audit_storage gzip --output_dir output/experiments/<weaker>/v8_300q
```

## 6. Analysis (offline, no credentials)

```
# per output dir
python scripts/run_experiments.py analyze --split held_out --output_dir <dir> \
    --max_total_tokens 200000 --max_tool_calls 30 --model <model>
python scripts/run_experiments.py revisions --split held_out --output_dir <dir> \
    --max_total_tokens 200000 --max_tool_calls 30 --model <model>

# paired team − single (gate items 1 and 2)
python scripts/compare_runs.py --split held_out \
    --arm single=output/experiments/gpt-6-luna/v8_300q/ordinary_single_agent.json \
    --arm team=output/experiments/gpt-6-luna/v8_300q/ordinary_multi_agent.json \
    --arm team_hetero=output/experiments/gpt-6-luna/v8_300q_hetero/ordinary_multi_agent.json \
    --output benchmarks/design-v8-$DATE/results.json

python scripts/record_design.py finish --record benchmarks/design-v8-$DATE \
    --report benchmarks/design-v8-$DATE/results.json
```

Gate (frozen in `design_v8.txt`): team > single **and** CI lower bound of
(team − single) > −2 pts **and** `transitions.wrong_to_right ≥ right_to_wrong`
for the team arm. Write `benchmarks/design-v8-$DATE/outcome.txt` with the
numbers either way, then fill the placeholders in `report.md`.

## 7. Confirmation — only if the gate passes

Protected questions = every id not in `split.json` pilot/dev/held_out, on a
patient not in `manifests/development_exposure.json` or
`docs/diagnosis_v8/audited_subject_ids.json`. Draw a family-stratified 300
(seed 5106) once, write it as a split manifest, run single + team once with
the section-2 settings, score once with `compare_runs.py`. No reruns, no
second sample.
