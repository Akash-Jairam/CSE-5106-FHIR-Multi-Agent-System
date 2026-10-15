# Hand-off: running and iterating on V8 from a personal machine

Everything on branch `experiments/diagnosis-v8` was built and unit-tested
offline (605 tests). Nothing has touched a model or a FHIR store. This is the
plan for a machine where you own the credentials. Companion files:
[`launch.md`](launch.md) (exact commands, flag-checked), [`design_v8.txt`](design_v8.txt)
(gate text to freeze), [`report.md`](report.md) (fill the TBDs).

## 0. Get the branch there

```bash
# on the work machine (one push, no execution of agents)
git push -u origin experiments/diagnosis-v8
# on the personal machine
git clone <repo> && cd CSE-5106-FHIR-Multi-Agent-System
git checkout experiments/diagnosis-v8
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests          # expect 605 passed, 2 skipped
```

## 1. Credentials (personal, not corporate)

| Need | How | Check |
|---|---|---|
| FHIR store | Own GCP project with the MIMIC-IV FHIR demo (`scripts/setup_fhir_store.sh` or README "Data Preparation"); `gcloud auth application-default login`; fill `FHIR_CONFIG` in `config.yml` | `python scripts/check_setup.py` → 100 patients |
| Model | Any OpenAI-compatible chat-completions API with **function calling and `usage`**. Simplest: an OpenAI key in `config.yml` (`OPENAI_API_KEY`). Alternatives: personal Azure OpenAI/Foundry via litellm `azure/<deployment>` env (`AZURE_API_BASE`, `AZURE_API_VERSION`, `AZURE_API_KEY` or `AZURE_AD_TOKEN`), or GitHub Models (fine-grained PAT with `models:read`, paid tier; `--base_url https://models.github.ai/inference`) | `python scripts/check_setup.py --model <m>` |

Model-name constraints in `utils/core_utils.py`: `get_model_provider` needs
the name to match `gpt-\d` / `o\d` / `gemini` / `qwen` / `meta-llama`;
`--base_url` forces the `openai` provider. `litellm==1.75.9` is pinned — if a
new model is not in its registry, cost comes back `null` (fine; the team's
runs also had null cost) but token counting still works. `gpt-6-luna` gets
`reasoning_effort: none` via `_is_luna_chat_model`; for another reasoning
model add the same `extra_body` branch there, otherwise tool calls can be
slow or truncated.

Pick **one** base model for everything and write it into `report.md` §2 as a
footnote; comparability with V2–V7 (gpt-6-luna) is lost if it differs, but
the within-run comparison (direct / single / team, same fingerprint) is what
the gate uses.

## 2. Smoke (not an experiment entry; throwaway output dirs)

```bash
python scripts/run_experiments.py run --split pilot --limit 5 --arms ordinary \
    --team_strategy multi_agent_v8 --audit_storage gzip --num_processes 2 --save_interval 1 \
    --model <m> --output_dir output/smoke/v8_pilot5
python scripts/run_experiments.py analyze --split pilot --limit 5 --arms ordinary \
    --model <m> --output_dir output/smoke/v8_pilot5
```

Look for, in `output/smoke/v8_pilot5/ordinary_multi_agent.json`:
`revision.protocol == "convention-verified-v8"`, a `release_policy`, zero
rows failing inside `convention_checks` (the checks must never raise; a
terminal error with a traceback mentioning `convention_checks.py` is a bug to
fix before anything else).

## 3. Iteration loop (where the real work is)

The checks were written from the audit without seeing live FHIR. Expect two
kinds of fixes, both allowed before the freeze and both changing the
fingerprint (so no partial results carry over):

**3a. Check calibration on real resources — no LLM needed.** Write
`scripts/validate_convention_checks_live.py`: for each of the 45 question
ids in `root_causes.csv` (dev patients; exposure already recorded), pull the
patient's resources through `fhir_client`, build a workspace like
`LedgerSolver` does, and report per check whether it *would* fire on the
gold-shaped answer and on the typical wrong answer described in the
`notes` column. Never print answers. Specific things to settle:
- T054: is `Specimen.type.coding[0].display` the string the gold expects?
- T055: do organism Observations hang off `hasMember` as assumed?
- T043: confirm `Encounter.priority` really collapses `admission_type`
  (if the raw string *is* recoverable somewhere, relabel those 4 rows from
  `label_issue` → `convention` and strengthen `admission_type_field`).
- T051/T052: tie sets at minute precision within the same encounter set.
- T001: does `complete_types()` ever contain `MedicationRequest` in practice?
  If the solver never retrieves the full closure, `distinct_count_scope` will
  flag every count and the repair must actually fetch it — watch the 6-tool
  repair cap.

**3b. Repair behaviour with the model.** Run the team alone on the core ids
(they are `held_out` ids, so this is exploratory dev use — note it in
`exposure.md`):

```bash
python - <<'EOF'
import csv; ids=[r['question_id'] for r in csv.DictReader(open('docs/diagnosis_v8/root_causes.csv'))]
open('/tmp/core_ids.txt','w').write('\n'.join(ids))
EOF
# run_agent.load_inputs(question_ids=...) exists; expose it with a small
# --question_ids_file flag on run_agent.py if you want this subset, or just use
# --split held_out --limit 60 for a cheaper first look.
```

Read `revision.convention_issues`, `routing`, `release_policy` and the repair
phase token/tool usage. Tune: which checks fire too often (false positives
cost a repair and risk a right→wrong flip), whether 45k/6 is enough, whether
`repair_message` wording gets the solver to fix the *field* rather than
re-derive the same answer. Keep the release rule conservative; the historic
failure mode (V2) is overwriting correct drafts.

Each loop: edit → `pytest -q tests/test_convention_checks.py
tests/test_convention_verified_agent.py` → re-run the subset. Commit as you
go; nothing here is frozen yet.

## 4. Freeze and run (once 3a/3b stop changing the code)

Follow `launch.md` §1–§5 verbatim. Order and rough cost at V7's token
profile (~115k team, ~72k single, ~10k direct per question):

| Step | Runs | ≈ tokens | Notes |
|---|---|---|---|
| freeze | — | — | `record_design.py freeze` with `design_v8.txt`; commit the record |
| C paired 300 | 900 | 60M | 4 workers took the team several hours per arm; expect 3–14 provider timeouts per arm — do **not** resume into the frozen dir |
| C' hetero verifier | 300 | 35M | pick a different family that passes `check_setup`; separate output dir |
| B1 recovery | 200 | 20M | `--arms recovery`, same dir as C |
| B2 weaker model | 900 | 40M | separate `output/experiments/<weaker>/` dir |

If budget forces a cut: C first (gate), then B1 (the only prior hint of a
team effect), then B2, then C'.

## 5. Analyse, gate, report

`launch.md` §6. Gate = team > single **and** CI lower bound > −2 pts **and**
`wrong_to_right ≥ right_to_wrong`. Write `benchmarks/design-v8-<date>/outcome.txt`
either way, replace the TBDs in `report.md` §2 and §5, regenerate the
accuracy-vs-tokens plot (`run_experiments.py analyze` writes it), commit.
Only on a pass: `launch.md` §7 confirmation, once.

## 6. Rules that keep the result reportable

- One `--output_dir` per code fingerprint × model × strategy; never mix.
- No `run` re-issue into a frozen directory (resume = retry = protocol break).
- The protected questions (not in `split.json`, patients not in
  `development_exposure.json` / `audited_subject_ids.json`) are touched only
  by the once-only confirmation.
- Anything learned from gold SQL or from reading answers on dev questions
  goes into `exposure.md`.
- Agents never see question ids, template ids, gold SQL or correctness.
- A negative V8 result plus B1/B2 and the taxonomy is a complete deliverable;
  don't start a V9 on the same 300.
