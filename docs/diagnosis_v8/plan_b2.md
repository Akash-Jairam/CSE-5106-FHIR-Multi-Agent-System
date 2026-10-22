# Plan B2: does the verified team help when the base model is weaker?

For the credentialed machine (the V9 checkout). Nine same-model team designs
and one protected-set confirmation did not beat a tool-equipped single agent
on gpt-6-luna (`report.md`, `plan_v9.md`). The one untested, proposal-stated
condition left is the base model. This plan runs it as a single prespecified
experiment, with an optional heterogeneous-verifier arm, and then stops.

## 1. Question and hypotheses (write these into `design_b2.txt` before any run)

**Question.** On a weaker base model, does the V9 verified team improve typed
accuracy over the single agent with the same tools and the same 200k / 30
ceiling?

- **H-B2a (headroom).** Direct LLM and single agent drop substantially on the
  weaker model (direct ≪ 69 %, single ≪ 82 %); the gap single − direct widens.
- **H-B2b (team gain).** Team − single on the weaker model > 0 with patient-block
  95 % CI lower bound > −2 pts, and the mechanism record shows the gain comes
  from `null_rule_recompute` / `adjudicated_recompute` W→R releases, not from
  primary run-to-run variance (team primary vs fresh single within ±2).
- **H-B2c (optional, heterogeneous).** Weak primary + luna recompute/adjudicator
  (same aggregate ceiling) beats the weak single by more than the same-model
  team does. Deviation from "same base LLM in three contexts"; labelled as such.

Pre-registered prediction (from run C / confirmation mechanics): the same-model
team gains ≤ +2 on luna; on a weaker model the single abstains and budget-stops
more often, so the null rule has more to work with. State a number before
launch (e.g. "+3 to +6 if H-B2b holds") and compare after.

**Verdict rule.** H-B2b supported ⇒ one confirmation on the protected
`confirmation_v9.json` sample with the weaker model (single + team, 600
attempts). Not supported ⇒ final negative; no further runs. H-B2c is
reported but never promoted without its own confirmation.

## 2. Pick the weaker model (one afternoon, cheap)

Criteria, in order: same endpoint and SDK as luna; supports tool calls and
`usage`; passes `scripts/check_setup.py --model <m>`; on the 5-question pilot
smoke all three arms run to completion with ≥ 1 correct answer each and no
schema errors; direct-LLM dev accuracy clearly below luna's 69 % (if it is
within 3 pts, the model is not "weaker" for this purpose — pick the next).

```
python scripts/check_setup.py --model <m>
python scripts/run_experiments.py run --split pilot --limit 5 --arms ordinary \
    --team_strategy multi_agent_v9 --model <m> \
    --max_total_tokens 200000 --max_tool_calls 30 --num_processes 1 --save_interval 1 \
    --audit_storage gzip --output_dir output/smoke/b2_<m>_pilot5
```

Candidates in order: the smallest "mini"/"nano" tier on the account, then the
previous-generation small model. Record the chosen model, the smoke counts and
the rejected candidates in `design_b2.txt`. Do not look at held_out questions
while choosing.

## 3. Freeze

```
DATE=$(date +%Y%m%d)
python scripts/record_design.py freeze --record benchmarks/design-b2-$DATE \
    --description docs/diagnosis_v8/design_b2.txt \
    --output_directory output/experiments/<m>/b2_300q \
    --model <m> --max_total_tokens 200000 --max_tool_calls 30 --questions 300 \
    --arm ordinary_direct_llm --arm ordinary_single_agent --arm ordinary_multi_agent \
    --command_text "see docs/diagnosis_v8/plan_b2.md section 4"
```

Commit `design.txt`, `manifest.json`, `source.zip`. No code or prompt change
after this; the guide, tools and controller are exactly the V9-confirmation
code (same `code_hash`).

## 4. Run (900 first attempts, dev `held_out`)

```
python scripts/run_experiments.py run --split held_out --arms ordinary \
    --team_strategy multi_agent_v9 --model <m> \
    --max_total_tokens 200000 --max_tool_calls 30 --num_processes 4 --save_interval 1 \
    --audit_storage gzip --output_dir output/experiments/<m>/b2_300q
```

Operational rules learned from V8/V9:
- Check the account has credits for ~1,000 attempts before starting; a
  mid-run `RateLimitError … no credits` makes a billing-retry deviation
  necessary. If it happens anyway, follow the confirmation precedent: decide
  and write the rule before scoring, keep the raw file, retry only the
  zero-usage/billing failures into a derived `*.billing_retried.json`, report
  both scores.
- Keep the machine awake (`caffeinate -i` on macOS) or latency is void.
- Score once, with the same `analyze` / `revisions` / `compare_runs.py`
  commands as V9 (`--team_strategy multi_agent_v9 --audit_storage gzip`).

## 5. Optional H-B2c arm (300 team attempts)

Only if H-B2b is *not* clearly supported and budget remains. Requires a flag
on `multi_agent_v9` that sets the recompute and adjudicator model separately
(the V8 `--verifier_model` plumbing is the template; the primary stays on
`<m>`, aggregate ceiling unchanged, tokens counted across both models).
Implement, unit-test, and record it in `design_b2.txt` **before** launching
the section-4 run, or run it as a separately frozen record
`design-b2het-$DATE`. Output dir `output/experiments/<m>/b2_300q_hetero`;
compare against the section-4 weak single, not a new control.

## 6. Analysis and reporting

Per arm and family: accuracy with CIs, budget stops, terminal errors, mean
tokens and observed cost (watch cache hit rate again), `release_policy`
counts, W→R / R→W by policy, team primary vs fresh single. Paired
team − single with patient-block bootstrap (seed 5106). Produce the
accuracy-vs-tokens plot with the luna arms on the same axes.

Report rows to add to `report.md` §2 / §5:

| Model | Direct | Single | Team (V9) | Δ team − single (95 % CI) | Controller net | Tokens team / single |
|---|---|---|---|---|---|---|
| gpt-6-luna (dev, run C/V9) | 208 | 244–250 | 241 / 249 | −1.7 … +5 (dev), −0.3 [−2.8, +1.9] (confirmation) | +1 | 1.5× |
| `<m>` (dev) | TBD | TBD | TBD | TBD | TBD | TBD |
| `<m>` + luna verifier (optional) | — | same | TBD | TBD | TBD | TBD |

Interpretation templates:
- Both H-B2a and H-B2b hold ⇒ "coordination helps when the base model lacks
  headroom; on a strong model the single agent is the better engineering
  choice." Run the confirmation before claiming it.
- H-B2a holds, H-B2b fails ⇒ "the failure is not about headroom; same-model
  verification cannot see convention errors at any capability level." Final.
- H-B2a fails ⇒ the model was not weak enough; report, do not iterate on
  model choice more than once.

## 7. Exposure and discipline

- Tenth use of the 300 dev questions; exploratory by definition. Only the
  weaker-model confirmation (section 1) is confirmatory.
- No prompt, guide, check or controller edits between freeze and scoring.
- Never print answers or `true_answer`; ids, templates, counts, policies only.
- Files to send back for write-up: `benchmarks/design-b2-*/`, the three arm
  JSONs (+ `.run.json`), `analysis/`, `revisions/`, and the smoke directory
  of the chosen model.

## Todos

1. `b2-model-select` — smoke candidates, pick `<m>`, record in `design_b2.txt`.
2. `b2-design-text` — write hypotheses, prediction, verdict rule, model choice.
3. `b2-freeze` — `record_design.py freeze`, commit.
4. `b2-run-dev` — 900 attempts, score once, analyze/revisions/compare.
5. `b2-hetero-optional` — flag + tests + separate freeze + 300 attempts (optional).
6. `b2-confirmation` — only if H-B2b holds: single + team on `confirmation_v9.json` with `<m>`.
7. `b2-report` — rows, plot, interpretation; close the project.

Dependencies: 2→1; 3→2; 4→3; 5→3; 6→4; 7→4 (and 5, 6 if run).
