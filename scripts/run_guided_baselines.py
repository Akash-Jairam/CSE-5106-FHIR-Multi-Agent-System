"""Guided baselines: how much of multi-agent v2's lead remains when the baselines get the analyst's instructions.

    python scripts/run_guided_baselines.py run --model gpt-oss:120b --base_url http://127.0.0.1:28431/v1 \
        --max_total_tokens 200000 --max_tool_calls 30
    python scripts/run_guided_baselines.py status  ... (same settings)
    python scripts/run_guided_baselines.py analyze ... (same settings)

On Compute2, submit through the job scripts (docs/results/guided-baselines.md):

    RUNNER=scripts/run_guided_baselines.py sbatch --job-name=guided-gpt-oss --dependency=singleton \
        scripts/run_experiments_vllm.sbatch
    RUNNER=scripts/run_guided_baselines.py OLLAMA_MODELS=/scratch2/fs1/di2cap/$USER/ollama-models \
        sbatch --job-name=guided-qwen3-32b scripts/run_experiments_qwen.sbatch

Five ordinary arms run in one batch: multi_agent, single_agent and direct_llm as they are, and

  single_agent_guided  SingleAgent whose system prompt also contains MultiTurnCodeResourceAgent.search_guidance
                       and SEMANTIC_GUIDANCE. It already has WORKED_EXAMPLES, so it then has every shared block
                       the analysis role's prompt has.
  direct_llm_guided    DirectLLMAgent whose system prompt also contains SEMANTIC_GUIDANCE. It has no tools, so
                       it does not get the search guidance, which is about filtering records in code.

Nothing else about the baselines changes. The guided prompts are set on the subclass instances and the module
constants are left alone, so multi_agent (whose packet reading reuses the direct prompt) is the system already
run, and agent/, utils/ and the code fingerprint are unchanged.

Everything else is scripts/run_experiments.py with these five arms: the same run_agent.py arguments, result
identities and resume, `plan`, `status`, and `analyze` (scripts/analysis.py with the comparisons below). `analyze`
also prints exact McNemar tests and failed first attempts by kind. Results go to
output/experiments/<model>/guided_<split> for manifests/split.json, else guided_<manifest name> (for example
guided_held_out_v2). Prints aggregates only, never answers.
"""
import collections
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run_agent  # noqa: E402
import utils.core_utils as core_utils  # noqa: E402
from agent.direct_llm_agent import DirectLLMAgent  # noqa: E402
from agent.ehr_semantics import SEMANTIC_GUIDANCE  # noqa: E402
from agent.multi_turn_code_resource_agent import MultiTurnCodeResourceAgent  # noqa: E402
from agent.output_schema import ANSWER_FORMAT_INSTRUCTIONS  # noqa: E402
from agent.single_agent import OUTPUT_GUIDANCE, SingleAgent  # noqa: E402
from scripts import analysis  # noqa: E402
from scripts import run_experiments as experiments  # noqa: E402
from scripts.experiment_stats import mcnemar_exact  # noqa: E402
from scripts.suite_breakdowns import _error_kind  # noqa: E402
from scripts.variability import first_attempt, is_failed  # noqa: E402

STRATEGIES = ("multi_agent", "single_agent", "direct_llm", "single_agent_guided", "direct_llm_guided")
# (label, arm a, arm b): a - b; the first is the primary comparison
COMPARISONS = [
    ("multi_agent - single_agent_guided", "ordinary_multi_agent", "ordinary_single_agent_guided"),
    ("single_agent_guided - single_agent", "ordinary_single_agent_guided", "ordinary_single_agent"),
    ("multi_agent - direct_llm_guided", "ordinary_multi_agent", "ordinary_direct_llm_guided"),
    ("direct_llm_guided - direct_llm", "ordinary_direct_llm_guided", "ordinary_direct_llm"),
    ("multi_agent - single_agent", "ordinary_multi_agent", "ordinary_single_agent"),
    ("multi_agent - direct_llm", "ordinary_multi_agent", "ordinary_direct_llm"),
]
SINGLE_GUIDANCE = (MultiTurnCodeResourceAgent.search_guidance, SEMANTIC_GUIDANCE)
DIRECT_GUIDANCE = (SEMANTIC_GUIDANCE,)


def with_guidance(prompt, tail, blocks):
    """`prompt` with the guidance blocks, verbatim, inserted before its closing `tail`."""
    if not prompt.endswith(tail):
        raise ValueError("The baseline prompt changed; update the guided baselines")
    return prompt[:-len(tail)] + "\n".join(blocks) + "\n" + tail


class SingleAgentGuided(SingleAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prompt = with_guidance(self.system_msg[0]["content"], OUTPUT_GUIDANCE, SINGLE_GUIDANCE)
        self.system_msg = [{"role": "system", "content": prompt}]


class DirectLLMGuided(DirectLLMAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prompt = with_guidance(self.system_msg["content"], ANSWER_FORMAT_INSTRUCTIONS, DIRECT_GUIDANCE)
        self.system_msg = {"role": "system", "content": prompt}


GUIDED_CLASSES = {"single_agent_guided": SingleAgentGuided, "direct_llm_guided": DirectLLMGuided}


def register():
    """Teach run_agent, the agent factory and the experiment runner the guided arms (inherited by forked workers)."""
    factory = core_utils.create_agent

    def create_agent(agent_strategy, model, verbose=False, base_url=None):
        if agent_strategy in GUIDED_CLASSES:
            return GUIDED_CLASSES[agent_strategy](model=model, verbose=verbose, base_url=base_url)
        return factory(agent_strategy, model, verbose, base_url)

    core_utils.create_agent = create_agent
    build_parser = run_agent.build_parser

    def parser_with_guided():
        parser = build_parser()
        for action in parser._actions:
            if action.dest == "agent_strategy":
                action.choices = list(action.choices) + list(GUIDED_CLASSES)
        return parser

    run_agent.build_parser = parser_with_guided
    experiments.ARMS = [experiments.Arm("ordinary", strategy) for strategy in STRATEGIES]
    experiments.EXPECTED = {"ordinary": len(STRATEGIES) * 300}
    analysis.COMPARISONS = COMPARISONS
    mp.set_start_method("fork", force=True)


def default_output_dir(args):
    manifest = os.path.splitext(os.path.basename(args.split_manifest))[0]
    leaf = "guided_" + (args.split if manifest == "split" else manifest)
    return os.path.join(ROOT, "output", "experiments", args.model.replace("/", "_"), leaf)


def supplement(args):
    """Exact McNemar tests and failed first attempts by kind, for the analyzable arms."""
    results = {arm.name: experiments.arm_results(arm, args, {}) for arm in experiments.ARMS}
    complete = {name: rows for name, (status, rows) in results.items() if status["analyzable"]}
    references = experiments.load_references(args, {qid for rows in complete.values() for qid in rows})
    correct = {name: {r["question_id"]: int(r["correct"]) for r in analysis.arm_records(rows, references)}
               for name, rows in complete.items()}
    print("\nExact McNemar tests (a - b; supplementary to the clustered intervals)")
    for label, a, b in COMPARISONS:
        if a in correct and b in correct:
            keys = sorted(correct[a])
            test = mcnemar_exact([correct[a][k] for k in keys], [correct[b][k] for k in keys])
            print(f"  {label}: only a {test['only_a']}, only b {test['only_b']} of {test['n']}, "
                  f"exact p = {test['p_value']:.2g}")
    print("\nFailed first attempts by kind")
    for name, rows in complete.items():
        attempts = [first_attempt(row) for row in rows.values()]
        failed = collections.Counter(_error_kind(str(row.get("error"))) for row in attempts if is_failed(row))
        print(f"  {name:<30} {sum(failed.values()):>3}/{len(attempts):<3} "
              + ", ".join(f"{kind} {n}" for kind, n in failed.most_common()))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("plan", "run", "status", "analyze"):
        print(__doc__)
        return 2
    register()
    if "--arms" not in argv:
        argv += ["--arms", "ordinary"]
    args = experiments.build_parser().parse_args(argv)
    if args.output_dir is None:
        argv += ["--output_dir", default_output_dir(args)]
        args = experiments.build_parser().parse_args(argv)
    code = experiments.main(argv)
    if argv[0] == "analyze":
        supplement(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
