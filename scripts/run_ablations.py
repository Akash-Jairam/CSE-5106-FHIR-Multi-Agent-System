"""Component ablations of the multi-agent system on a frozen subset of the generated questions.

    python scripts/run_ablations.py subset
    python scripts/run_ablations.py run --model gpt-oss:120b --base_url http://127.0.0.1:28431/v1 \
        --max_total_tokens 200000 --max_tool_calls 30
    python scripts/run_ablations.py analyze --model gpt-oss:120b --base_url ... (same settings)

`subset` draws 100 of the 300 held-out questions of the `generated` suite, stratified by polarity,
and freezes them in manifests/generated/fhir_native.ablation_subset.json. `run` runs every variant
on them; on Compute2, submit it through the vLLM job:

    RUNNER=scripts/run_ablations.py sbatch --job-name=ablations-gpt-oss scripts/run_experiments_vllm.sbatch

Variants are subclasses of MultiAgentSystem defined here and registered with the agent factory at
startup, so agent/, utils/ and the code fingerprint are unchanged and the suite results stay current.
Each differs from the full system in one component:

  full          the system as run in the suites (the control, rerun in the same batch)
  no_prefetch   retrieval does not prefetch the complete records; it fetches only what it asks for
  no_packet     analysis is not shown the starting packet (the packet reading still votes)
  no_guidance   analysis prompt without the search guidance, EHR conventions and worked examples
  no_verifier   verification never runs: no repair and no verifier vote

`analyze` scores each variant against `full` (first attempt, typed scorer, failed runs wrong) with
patient-clustered intervals and exact McNemar tests. It also scores, post hoc, each candidate the full
system's release vote chose from (draft, repaired answer, packet reading, verifier recomputation), and the
draft-then-repair answer without a vote, on the same questions and on the 300 questions of the `generated` suite. Prints aggregates only, never answers.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
from argparse import Namespace
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run_agent  # noqa: E402
import utils.core_utils as core_utils  # noqa: E402
from agent import analysis_agent  # noqa: E402
from agent.analysis_agent import AnalysisAgent  # noqa: E402
from agent.ehr_semantics import SEMANTIC_GUIDANCE  # noqa: E402
from agent.multi_agent import MultiAgentSystem  # noqa: E402
from agent.multi_turn_code_resource_agent import WORKED_EXAMPLES, MultiTurnCodeResourceAgent  # noqa: E402
from agent.retrieval_agent import RetrievalAgent  # noqa: E402
from agent.verification_agent import VerificationAgent  # noqa: E402
from scripts.experiment_stats import clustered_mean, mcnemar_exact, paired_difference  # noqa: E402
from scripts.run_experiments import load_references  # noqa: E402
from scripts.variability import first_attempt, is_failed, score_row  # noqa: E402
from utils.answer_scorer import score_answer  # noqa: E402

INPUT = os.path.join(ROOT, "final_dataset", "generated", "fhir_native.csv")
SPLIT_MANIFEST = os.path.join(ROOT, "manifests", "generated", "fhir_native.split.json")
FAMILIES = os.path.join(ROOT, "manifests", "generated", "fhir_native.families.csv")
SUBSET = os.path.join(ROOT, "manifests", "generated", "fhir_native.ablation_subset.json")
SPLIT = "held_out"
SIZE = 100
SEED = 5106
VARIANTS = ("full", "no_prefetch", "no_packet", "no_guidance", "no_verifier")
VOTERS = ("draft", "repaired", "direct", "verifier")


class NoPrefetch(MultiAgentSystem):
    def build_roles(self, context, workspace):
        roles = super().build_roles(context, workspace)
        self._retrieval = RetrievalAgent(self, context, workspace, max_turns=4, prefetch=False, direct=True)
        return {**roles, "retrieval": self._retrieval,
                "analysis": AnalysisAgent(self, context, workspace, packet=lambda: self._retrieval.packet)}


class NoPacket(MultiAgentSystem):
    def build_roles(self, context, workspace):
        return {**super().build_roles(context, workspace), "analysis": AnalysisAgent(self, context, workspace)}


def _without_guidance(prompt):
    for block in (MultiTurnCodeResourceAgent.search_guidance, SEMANTIC_GUIDANCE, WORKED_EXAMPLES):
        if block not in prompt:
            raise ValueError("The analysis prompt changed; update the no_guidance ablation")
        prompt = prompt.replace(block, "")
    return prompt


class NoGuidance(MultiAgentSystem):
    def build_roles(self, context, workspace):
        roles = super().build_roles(context, workspace)
        roles["analysis"].role.system_prompt = _without_guidance(analysis_agent.SYSTEM_PROMPT)
        return roles


class NoVerifier(MultiAgentSystem):
    def _verify(self, draft):
        return []


STRATEGIES = {"full": ("multi_agent", None), "no_prefetch": ("multi_agent_no_prefetch", NoPrefetch),
              "no_packet": ("multi_agent_no_packet", NoPacket),
              "no_guidance": ("multi_agent_no_guidance", NoGuidance),
              "no_verifier": ("multi_agent_no_verifier", NoVerifier)}
ABLATION_CLASSES = {strategy: cls for strategy, cls in STRATEGIES.values() if cls is not None}


def register():
    """Teach run_agent and the agent factory the variant strategies (inherited by forked workers)."""
    factory = core_utils.create_agent

    def create_agent(agent_strategy, model, verbose=False, base_url=None):
        if agent_strategy in ABLATION_CLASSES:
            return ABLATION_CLASSES[agent_strategy](model=model, verbose=verbose, base_url=base_url)
        return factory(agent_strategy, model, verbose, base_url)

    core_utils.create_agent = create_agent
    build_parser = run_agent.build_parser

    def parser_with_variants():
        parser = build_parser()
        for action in parser._actions:
            if action.dest == "agent_strategy":
                action.choices = list(action.choices) + list(ABLATION_CLASSES)
        return parser

    run_agent.build_parser = parser_with_variants
    mp.set_start_method("fork", force=True)


def output_path(args, variant):
    return os.path.join(args.output_dir, f"{variant}.json")


def runner_argv(args, variant):
    argv = ["--agent_strategy", STRATEGIES[variant][0], "--model", args.model, "--input", INPUT,
            "--output", output_path(args, variant), "--split_manifest", SPLIT_MANIFEST, "--split", SPLIT,
            "--subset", SUBSET, "--enable_cache", "true", "--num_processes", str(args.num_processes),
            "--save_interval", "10", "--quiet"]
    for key in ("base_url", "max_total_tokens", "max_tool_calls"):
        if getattr(args, key) is not None:
            argv += [f"--{key}", str(getattr(args, key))]
    return argv


def build_subset():
    import pandas as pd

    with open(SPLIT_MANIFEST, encoding="utf-8") as stream:
        held_out = json.load(stream)["splits"][SPLIT]["question_ids"]
    polarity = pd.read_csv(INPUT, usecols=["question_id", "polarity"]).set_index("question_id")["polarity"]
    rng = random.Random(SEED)
    chosen = set()
    for value in sorted(set(polarity[held_out])):
        stratum = [q for q in held_out if polarity[q] == value]
        chosen.update(rng.sample(stratum, round(SIZE * len(stratum) / len(held_out))))
    question_ids = [q for q in held_out if q in chosen]
    subset = {"version": 1, "split": SPLIT, "seed": SEED, "size": len(question_ids),
              "split_sha256": hashlib.sha256(Path(SPLIT_MANIFEST).read_bytes()).hexdigest(),
              "polarity_counts": {v: int((polarity[question_ids] == v).sum()) for v in sorted(set(polarity))},
              "question_ids": question_ids}
    with open(SUBSET, "w", encoding="utf-8") as stream:
        json.dump(subset, stream, indent=1)
        stream.write("\n")
    print(f"Wrote {os.path.relpath(SUBSET, ROOT)}: {subset['size']} questions, {subset['polarity_counts']}")


def run(args):
    register()
    failed = 0
    for variant in args.variants:
        print(f"== {variant}", flush=True)
        failed |= run_agent.main(runner_argv(args, variant)) or 0
    return failed


def _rows(path):
    with open(path, encoding="utf-8") as stream:
        return {row["question_id"]: first_attempt(row) for row in json.load(stream)}


def status(args):
    register()
    for variant in args.variants:
        settings = run_agent.build_parser().parse_args(runner_argv(args, variant))
        inputs = run_agent.load_inputs(INPUT, None, SPLIT_MANIFEST, SPLIT,
                                       question_ids=run_agent.load_subset(SUBSET, SPLIT_MANIFEST, SPLIT))
        path = settings.output
        if not os.path.exists(path):
            print(f"  {variant:<12} not started")
            continue
        with open(path + ".run.json", encoding="utf-8") as stream:
            current = json.load(stream) == run_agent.run_identity(settings, inputs)
        rows = _rows(path)
        done = sum(run_agent.is_done(row) for row in rows.values())
        print(f"  {variant:<12} {done}/{len(inputs)} done, {sum(is_failed(r) for r in rows.values())} failed"
              + ("" if current else ", STALE: different inputs, settings or code"))


def _rate(values, patients):
    s = clustered_mean(values, patients)
    return f"{s['estimate']:.1%} [{s['ci_low']:.1%}, {s['ci_high']:.1%}]"


def _difference(a, b, patients):
    d, t = paired_difference(a, b, patients), mcnemar_exact(a, b)
    return (f"{d['estimate'] * 100:+.1f} [{d['ci_low'] * 100:+.1f}, {d['ci_high'] * 100:+.1f}], "
            f"discordant {t['only_a']}/{t['only_b']}, exact p = {t['p_value']:.2g}")


def _candidate_scores(rows, references):
    """{system: {question_id: 0/1 or None}}: the released answer, each voter alone (None where it produced no
    candidate; an abstention is a candidate), and `draft_then_repair`, the repaired answer if there is one,
    else the draft (no vote)."""
    scores = {name: {} for name in ("released", *VOTERS, "draft_then_repair")}
    for qid, row in rows.items():
        reference = references[qid]
        scores["released"][qid] = score_row(row, reference)
        revision = row.get("revision")
        revision = json.loads(revision) if isinstance(revision, str) else revision or {}
        candidates = {} if is_failed(row) else (revision.get("release") or {}).get("candidates") or {}

        def score(answer):
            return score_answer(answer, reference["true_answer"], reference["question"],
                                reference.get("template", ""))["typed_correct"]

        for voter in VOTERS:
            scores[voter][qid] = score(candidates[voter]) if voter in candidates else None
        chosen = next((candidates[v] for v in ("repaired", "draft") if v in candidates), None)
        scores["draft_then_repair"][qid] = score(chosen) if "draft" in candidates else 0
    return scores


def _post_hoc(label, path, references):
    rows = _rows(path)
    scores = _candidate_scores(rows, references)
    qids = sorted(rows)
    print(f"\n  {label} (n={len(qids)})")
    for name in ("released", "draft", "draft_then_repair"):
        values = [0 if scores[name][q] is None else scores[name][q] for q in qids]
        patients = [references[q]["patient"] for q in qids]
        line = f"    {name:<18} {_rate(values, patients)}"
        if name != "released":
            line += f"; released - {name}: " + _difference([scores["released"][q] for q in qids], values, patients)
        print(line)
    for voter in VOTERS:
        answered = [q for q in qids if scores[voter][q] is not None]
        right = sum(scores[voter][q] for q in answered)
        agree = sum(scores[voter][q] == scores["released"][q] for q in answered)
        print(f"    {voter:<18} produced {len(answered)}/{len(qids)}; right {right}/{len(answered)} "
              f"where produced; same verdict as released {agree}/{len(answered)}")


def analyze(args):
    register()
    subset = run_agent.load_subset(SUBSET, SPLIT_MANIFEST, SPLIT)
    inputs = Namespace(input=INPUT, split_manifest=SPLIT_MANIFEST, families=FAMILIES)
    references = load_references(inputs, subset)
    qids = sorted(subset)
    patients = [references[q]["patient"] for q in qids]
    rows = {v: _rows(output_path(args, v)) for v in args.variants if os.path.exists(output_path(args, v))}
    print(f"== Ablations on {len(qids)} generated questions, {len(set(patients))} patients")
    full = [score_row(rows["full"][q], references[q]) for q in qids] if "full" in rows else None
    for variant, by_qid in rows.items():
        values = [score_row(by_qid[q], references[q]) for q in qids]
        tokens = sum((by_qid[q].get("tokens") or {}).get("total") or 0 for q in qids) / len(qids)
        line = f"  {variant:<12} {_rate(values, patients)}, mean tokens {tokens / 1000:.1f}k"
        if variant != "full" and full is not None:
            line += f"\n               {variant} - full: {_difference(values, full, patients)}"
        print(line)
    print("\n== Release vote, post hoc: each candidate the full system chose from")
    if "full" in rows:
        _post_hoc("ablation subset, full", output_path(args, "full"), references)
    suite = os.path.join(os.path.dirname(args.output_dir), "suite_generated", "ordinary_multi_agent.json")
    if os.path.exists(suite):
        suite_rows = _rows(suite)
        _post_hoc("generated suite", suite, load_references(inputs, list(suite_rows)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["subset", "run", "status", "analyze"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--max_total_tokens", type=int, default=None)
    parser.add_argument("--max_tool_calls", type=int, default=None)
    parser.add_argument("--num_processes", type=int, default=-1)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--output_dir", default=None, help="Default: output/experiments/<model>/ablation_generated")
    parser.add_argument("--split", default=SPLIT, choices=[SPLIT], help="Accepted for the vLLM job script")
    args = parser.parse_args(argv)
    if args.command == "subset":
        return build_subset()
    if not args.model:
        parser.error("--model is required")
    args.output_dir = args.output_dir or os.path.join(ROOT, "output", "experiments", args.model.replace("/", "_"),
                                                      "ablation_generated")
    return {"run": run, "status": status, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
