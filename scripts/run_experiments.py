"""Run the Phase 3 experiment (P3-19): 900 ordinary and 200 recovery runs.

Ordinary runs: each of the three systems answers all 300 held-out questions
(3 x 300 = 900). Recovery runs: the two tool-equipped systems answer the 50
perturbed held-out questions twice, from the intact and from the omitted
starting packet (2 x 2 x 50 = 200). The direct baseline has no tools, so it
cannot recover a missing record and has no recovery runs. The intact recovery
runs repeat the ordinary condition on the same questions, so intact vs
omitted is compared within one batch of runs.

Each arm is one run_agent.py call with its own result file. All arms share
one model, one base URL and one budget, and the tool cache is always on.
The runner binds each file to its inputs, settings and code, saves every
--save_interval questions, and skips completed questions, so rerunning the
same command resumes an interrupted experiment and retries failed questions.
Code changes make existing files stale; use a new --output_dir.

Usage (from the repository root):
    python scripts/run_experiments.py plan   --model gpt-5-nano
    python scripts/run_experiments.py run    --model gpt-5-nano --max_total_tokens 200000 --max_tool_calls 30
    python scripts/run_experiments.py status --model gpt-5-nano --max_total_tokens 200000 --max_tool_calls 30

`status` makes no model or FHIR calls and prints counts only, never answers.
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import run_agent  # noqa: E402

SYSTEMS = ("direct_llm", "single_agent", "multi_agent")
RECOVERY_SYSTEMS = ("single_agent", "multi_agent")
HELD_OUT = "held_out"
EXPECTED = {"ordinary": 900, "recovery": 200}
DEFAULT_INPUT = os.path.join(ROOT, "final_dataset", "questions_answers_sql_fhir.csv")
DEFAULT_SPLIT = os.path.join(ROOT, "manifests", "split.json")
DEFAULT_PERTURBATIONS = os.path.join(ROOT, "manifests", "perturbations.json")


@dataclass(frozen=True)
class Arm:
    group: str                # "ordinary" or "recovery"
    strategy: str
    condition: str = None     # recovery only: "intact" or "omitted"

    @property
    def name(self):
        return "_".join(part for part in (self.group, self.strategy, self.condition) if part)


ARMS = ([Arm("ordinary", strategy) for strategy in SYSTEMS]
        + [Arm("recovery", strategy, condition) for strategy in RECOVERY_SYSTEMS for condition in run_agent.CONDITIONS])


def build_parser():
    parser = argparse.ArgumentParser(description="Run, resume, or check the Phase 3 experiment.")
    parser.add_argument("command", choices=["plan", "run", "status"])
    parser.add_argument("--model", required=True, help="One base model shared by every system")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--max_total_tokens", type=int, default=None, help="Per-question ceiling for every system")
    parser.add_argument("--max_tool_calls", type=int, default=None, help="Per-question ceiling for every system")
    parser.add_argument("--arms", nargs="+", default=["ordinary", "recovery"],
                        help="Groups (ordinary, recovery) or arm names, e.g. recovery_multi_agent_omitted")
    parser.add_argument("--split", choices=["pilot", "dev", HELD_OUT], default=HELD_OUT,
                        help="pilot/dev for smoke runs; recovery arms need held_out")
    parser.add_argument("--limit", type=int, default=None, help="First N questions per arm, for smoke runs")
    parser.add_argument("--output_dir", default=None,
                        help="Default: output/experiments/<model>/<split>[_limit<N>]")
    parser.add_argument("--num_processes", type=int, default=-1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--split_manifest", default=DEFAULT_SPLIT)
    parser.add_argument("--perturbations", default=DEFAULT_PERTURBATIONS)
    parser.add_argument("--verbose", action="store_true", help="Print each agent's progress")
    return parser


def select_arms(names, split):
    known = {arm.name for arm in ARMS} | {arm.group for arm in ARMS}
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"Unknown arms: {', '.join(unknown)}; choose from {', '.join(sorted(known))}")
    arms = [arm for arm in ARMS if arm.group in names or arm.name in names]
    if split != HELD_OUT and any(arm.group == "recovery" for arm in arms):
        raise ValueError("Recovery arms use the held-out perturbations; drop them or use --split held_out")
    return arms


def output_dir(args):
    if args.output_dir:
        return args.output_dir
    leaf = args.split + (f"_limit{args.limit}" if args.limit is not None else "")
    return os.path.join(ROOT, "output", "experiments", args.model.replace("/", "_"), leaf)


def runner_argv(arm, args):
    """The run_agent.py arguments for one arm."""
    argv = ["--agent_strategy", arm.strategy, "--model", args.model,
            "--input", args.input, "--output", os.path.join(output_dir(args), f"{arm.name}.json"),
            "--split_manifest", args.split_manifest, "--split", args.split,
            "--enable_cache", "true", "--num_processes", str(args.num_processes),
            "--save_interval", str(args.save_interval)]
    for key in ("base_url", "max_total_tokens", "max_tool_calls", "limit"):
        if getattr(args, key) is not None:
            argv += [f"--{key}", str(getattr(args, key))]
    if arm.condition:
        argv += ["--perturbations", args.perturbations, "--condition", arm.condition]
    return argv + ([] if args.verbose else ["--quiet"])


def arm_inputs(arm, settings, perturbations):
    question_ids = list(perturbations) if arm.condition else None
    return run_agent.load_inputs(settings.input, settings.limit, settings.split_manifest, settings.split,
                                 question_ids=question_ids)


def arm_status(arm, args, perturbations):
    """Counts for one arm's result file; never reads or prints answers."""
    settings = run_agent.build_parser().parse_args(runner_argv(arm, args))
    inputs = arm_inputs(arm, settings, perturbations)
    status = {"arm": arm.name, "group": arm.group, "questions": len(inputs), "output": settings.output,
              "completed": 0, "failed": 0, "state": "not started"}
    if not os.path.exists(settings.output):
        return status
    identity_file = settings.output + ".run.json"
    saved = None
    if os.path.exists(identity_file):
        with open(identity_file, encoding="utf-8") as stream:
            saved = json.load(stream)
    if saved != run_agent.run_identity(settings, inputs):
        status["state"] = "stale: different inputs, settings or code; use a new --output_dir"
        return status
    with open(settings.output, encoding="utf-8") as stream:
        rows = {row.get("question_id"): row for row in json.load(stream)}
    rows = [rows.get(qid, {}) for qid in inputs["question_id"]]
    status["completed"] = sum(run_agent.is_done(row) for row in rows)
    status["failed"] = sum(not run_agent.is_done(row) and bool(row.get("error")) for row in rows)
    status["state"] = "complete" if status["completed"] == len(inputs) else "in progress"
    return status


def load_recovery_set(arms, args):
    if not any(arm.condition for arm in arms):
        return {}
    from tools.perturbation import load_perturbations
    return load_perturbations(args.perturbations)


def print_status(statuses):
    width = max(len(s["arm"]) for s in statuses)
    print(f"{'arm':<{width}}  {'done':>5} {'failed':>6} {'total':>5}  state")
    for s in statuses:
        print(f"{s['arm']:<{width}}  {s['completed']:>5} {s['failed']:>6} {s['questions']:>5}  {s['state']}")
    for group in EXPECTED:
        members = [s for s in statuses if s["group"] == group]
        if members:
            done, total = sum(s["completed"] for s in members), sum(s["questions"] for s in members)
            print(f"{group}: {done}/{total} runs complete")


def check_sizes(statuses, arms, args):
    """The full experiment must be exactly 900 ordinary and 200 recovery runs."""
    if args.split != HELD_OUT or args.limit is not None:
        return
    for group, expected in EXPECTED.items():
        if all(arm in arms for arm in ARMS if arm.group == group):
            total = sum(s["questions"] for s in statuses if s["group"] == group)
            if total != expected:
                raise ValueError(f"The {group} arms have {total} runs; the experiment needs {expected}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for key in ("max_total_tokens", "max_tool_calls", "limit"):
        if getattr(args, key) is not None and getattr(args, key) < 0:
            parser.error(f"--{key} must be non-negative")
    try:
        arms = select_arms(args.arms, args.split)
    except ValueError as exc:
        parser.error(str(exc))
    perturbations = load_recovery_set(arms, args)
    statuses = [arm_status(arm, args, perturbations) for arm in arms]
    check_sizes(statuses, arms, args)

    if args.command == "plan":
        print(f"Model: {args.model}; split: {args.split}; output: {output_dir(args)}")
        for arm, status in zip(arms, statuses):
            print(f"\n{arm.name}: {status['questions']} questions")
            shown = [os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else a for a in runner_argv(arm, args)]
            print("  python run_agent.py " + " ".join(shown))
        print()
        print_status(statuses)
        return 0
    if args.command == "status":
        print_status(statuses)
        return 0

    failures = []
    for arm in arms:
        print(f"\n=== {arm.name} ===", flush=True)
        try:
            code = run_agent.main(runner_argv(arm, args))
        except Exception as exc:  # one arm's failure must not stop the others
            code = f"{type(exc).__name__}: {exc}"
        if code != 0:
            failures.append((arm.name, code))
            print(f"{arm.name} stopped: {code}", flush=True)
    print()
    statuses = [arm_status(arm, args, perturbations) for arm in arms]
    print_status(statuses)
    for name, code in failures:
        print(f"FAILED {name}: {code}")
    incomplete = any(s["state"] != "complete" for s in statuses)
    if incomplete:
        print("Rerun the same command to resume and retry failed questions.")
    return 1 if failures or incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
