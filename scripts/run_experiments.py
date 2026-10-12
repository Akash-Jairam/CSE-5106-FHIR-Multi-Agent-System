"""Run the Phase 3 experiment: 900 ordinary, 200 recovery (P3-19) and 300 variability (P3-20) runs.

Ordinary runs: each of the three systems answers all 300 held-out questions
(3 x 300 = 900). Recovery runs: the two tool-equipped systems answer the 50
perturbed held-out questions twice, from the intact and from the omitted
starting packet (2 x 2 x 50 = 200). The direct baseline has no tools, so it
cannot recover a missing record and has no recovery runs. The intact recovery
runs repeat the ordinary condition on the same questions, so intact vs
omitted is compared within one batch of runs. Variability runs: each system
answers a fixed 20-question held-out subset (manifests/variability_subset.json)
5 times (3 x 5 x 20 = 300); `variability` reports how much results change
between repeats.

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

    python scripts/run_experiments.py variability --model gpt-5-nano --max_total_tokens 200000 --max_tool_calls 30

`status` and `variability` make no model or FHIR calls and print aggregates only, never answers.
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
HELD_OUT_GROUPS = ("recovery", "variability")
REPEATS = 5
EXPECTED = {"ordinary": 900, "recovery": 200, "variability": 300}
DEFAULT_INPUT = os.path.join(ROOT, "final_dataset", "questions_answers_sql_fhir.csv")
DEFAULT_SPLIT = os.path.join(ROOT, "manifests", "split.json")
DEFAULT_PERTURBATIONS = os.path.join(ROOT, "manifests", "perturbations.json")
DEFAULT_SUBSET = os.path.join(ROOT, "manifests", "variability_subset.json")
DEFAULT_FAMILIES = os.path.join(ROOT, "manifests", "question_families.csv")


@dataclass(frozen=True)
class Arm:
    group: str                # "ordinary", "recovery" or "variability"
    strategy: str
    condition: str = None     # recovery only: "intact" or "omitted"
    repeat: int = None        # variability only: 1..REPEATS

    @property
    def name(self):
        repeat = f"r{self.repeat}" if self.repeat else None
        return "_".join(part for part in (self.group, self.strategy, self.condition, repeat) if part)


ARMS = ([Arm("ordinary", strategy) for strategy in SYSTEMS]
        + [Arm("recovery", strategy, condition) for strategy in RECOVERY_SYSTEMS for condition in run_agent.CONDITIONS]
        + [Arm("variability", strategy, repeat=r) for strategy in SYSTEMS for r in range(1, REPEATS + 1)])


def build_parser():
    parser = argparse.ArgumentParser(description="Run, resume, or check the Phase 3 experiment.")
    parser.add_argument("command", choices=["plan", "run", "status", "variability"])
    parser.add_argument("--model", required=True, help="One base model shared by every system")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--max_total_tokens", type=int, default=None, help="Per-question ceiling for every system")
    parser.add_argument("--max_tool_calls", type=int, default=None, help="Per-question ceiling for every system")
    parser.add_argument("--arms", nargs="+", default=None,
                        help="Groups (ordinary, recovery, variability) or arm names, e.g. variability_multi_agent_r2. "
                             "Default: every group (only variability for the variability command)")
    parser.add_argument("--split", choices=["pilot", "dev", HELD_OUT], default=HELD_OUT,
                        help="pilot/dev for smoke runs; recovery and variability arms need held_out")
    parser.add_argument("--limit", type=int, default=None, help="First N questions per arm, for smoke runs")
    parser.add_argument("--output_dir", default=None,
                        help="Default: output/experiments/<model>/<split>[_limit<N>]")
    parser.add_argument("--num_processes", type=int, default=-1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--split_manifest", default=DEFAULT_SPLIT)
    parser.add_argument("--perturbations", default=DEFAULT_PERTURBATIONS)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--families", default=DEFAULT_FAMILIES, help="Templates for the typed scorer")
    parser.add_argument("--verbose", action="store_true", help="Print each agent's progress")
    return parser


def select_arms(names, split):
    known = {arm.name for arm in ARMS} | {arm.group for arm in ARMS}
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"Unknown arms: {', '.join(unknown)}; choose from {', '.join(sorted(known))}")
    arms = [arm for arm in ARMS if arm.group in names or arm.name in names]
    if split != HELD_OUT and any(arm.group in HELD_OUT_GROUPS for arm in arms):
        raise ValueError("Recovery and variability arms use frozen held-out questions; "
                         "drop them or use --split held_out")
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
    if arm.repeat:
        argv += ["--subset", args.subset, "--repeat", str(arm.repeat)]
    return argv + ([] if args.verbose else ["--quiet"])


def arm_inputs(arm, settings, question_sets):
    """The arm's questions; `question_sets` maps the recovery and variability groups to their frozen IDs."""
    return run_agent.load_inputs(settings.input, settings.limit, settings.split_manifest, settings.split,
                                 question_ids=question_sets.get(arm.group))


def arm_results(arm, args, question_sets):
    """(status, {question_id: row}) for one arm; rows are empty unless the file is current."""
    settings = run_agent.build_parser().parse_args(runner_argv(arm, args))
    inputs = arm_inputs(arm, settings, question_sets)
    status = {"arm": arm.name, "group": arm.group, "questions": len(inputs), "output": settings.output,
              "completed": 0, "failed": 0, "state": "not started"}
    if not os.path.exists(settings.output):
        return status, {}
    identity_file = settings.output + ".run.json"
    saved = None
    if os.path.exists(identity_file):
        with open(identity_file, encoding="utf-8") as stream:
            saved = json.load(stream)
    if saved != run_agent.run_identity(settings, inputs):
        status["state"] = "stale: different inputs, settings or code; use a new --output_dir"
        return status, {}
    with open(settings.output, encoding="utf-8") as stream:
        saved_rows = {row.get("question_id"): row for row in json.load(stream)}
    rows = {qid: saved_rows.get(qid, {}) for qid in inputs["question_id"]}
    status["completed"] = sum(run_agent.is_done(row) for row in rows.values())
    status["failed"] = sum(not run_agent.is_done(row) and bool(row.get("error")) for row in rows.values())
    status["state"] = "complete" if status["completed"] == len(inputs) else "in progress"
    return status, rows


def arm_status(arm, args, question_sets):
    """Counts for one arm's result file; never prints answers."""
    return arm_results(arm, args, question_sets)[0]


def load_question_sets(arms, args):
    """Frozen question IDs for the selected held-out groups, validated before any run."""
    groups = {arm.group for arm in arms}
    question_sets = {}
    if "recovery" in groups:
        from tools.perturbation import load_perturbations
        question_sets["recovery"] = list(load_perturbations(args.perturbations))
    if "variability" in groups:
        question_sets["variability"] = run_agent.load_subset(args.subset, args.split_manifest, args.split)
        with open(args.subset, encoding="utf-8") as stream:
            repeats = json.load(stream).get("repeats")
        if repeats != REPEATS:
            raise ValueError(f"The subset manifest plans {repeats} repeats; the experiment runs {REPEATS}")
    return question_sets


def variability_report(arms, args, question_sets):
    """Per-system spread over the complete repeats of the variability arms."""
    import pandas as pd
    from scripts.variability import summarize

    question_ids = question_sets["variability"]
    frame = run_agent.read_input_data(args.input)
    frame = frame[frame["question_id"].isin(question_ids)]
    templates = pd.read_csv(args.families, usecols=["question_id", "template"]).set_index("question_id")["template"]
    references = {row.question_id: {"question": row.question, "true_answer": row.true_answer,
                                    "template": templates.get(row.question_id, "")}
                  for row in frame.itertuples()}
    report = {"subset": os.path.relpath(args.subset, ROOT), "planned_repeats": REPEATS, "systems": {}}
    for strategy in SYSTEMS:
        complete, skipped = [], []
        for arm in (a for a in arms if a.group == "variability" and a.strategy == strategy):
            status, rows = arm_results(arm, args, question_sets)
            (complete if status["state"] == "complete" else skipped).append((arm, rows))
        if complete:
            asked = {qid: references[qid] for qid in complete[0][1]}  # the subset, after any --limit
            report["systems"][strategy] = {**summarize([rows for _, rows in complete], asked),
                                           "skipped_incomplete": [arm.name for arm, _ in skipped]}
    return report


def print_variability(report):
    def fmt(spread, scale=1, digits=3):
        if spread["mean"] is None:
            return "n/a"
        sd = "n/a" if spread["sd"] is None else f"{spread['sd'] * scale:.{digits}f}"
        return (f"{spread['mean'] * scale:.{digits}f} (sd {sd}, "
                f"{spread['min'] * scale:.{digits}f}-{spread['max'] * scale:.{digits}f})")

    if not report["systems"]:
        print("No complete variability repeats yet.")
    for strategy, s in report["systems"].items():
        print(f"\n{strategy}: {s['repeats']}/{report['planned_repeats']} repeats x {s['questions']} questions")
        print(f"  accuracy          {fmt(s['accuracy'])}")
        print(f"  abstention rate   {fmt(s['abstention_rate'])}")
        print(f"  mean tokens       {fmt(s['mean_tokens'], digits=0)}")
        print(f"  median latency s  {fmt(s['median_latency'], digits=2)}")
        print(f"  questions always/never/sometimes correct: "
              f"{s['always_correct']}/{s['never_correct']}/{s['unstable']}")
        if s["correctness_agreement"] is not None:
            print(f"  pairwise correctness agreement {s['correctness_agreement']:.3f}; "
                  f"identical answers in every repeat {s['identical_answers']:.3f}")
        if s["skipped_incomplete"]:
            print(f"  skipped incomplete: {', '.join(s['skipped_incomplete'])}")


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
    """The full experiment must be exactly 900 ordinary, 200 recovery and 300 variability runs."""
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
    names = args.arms or (["variability"] if args.command == "variability" else list(EXPECTED))
    try:
        arms = select_arms(names, args.split)
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "variability" and not any(arm.group == "variability" for arm in arms):
        parser.error("Select variability arms for the variability report")
    question_sets = load_question_sets(arms, args)
    statuses = [arm_status(arm, args, question_sets) for arm in arms]
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
    if args.command == "variability":
        report = variability_report(arms, args, question_sets)
        os.makedirs(output_dir(args), exist_ok=True)
        path = os.path.join(output_dir(args), "variability_report.json")
        run_agent._write_json(report, path)
        print_variability(report)
        print(f"\nWrote {path}")
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
    statuses = [arm_status(arm, args, question_sets) for arm in arms]
    print_status(statuses)
    for name, code in failures:
        print(f"FAILED {name}: {code}")
    incomplete = any(s["state"] != "complete" for s in statuses)
    if incomplete:
        print("Rerun the same command to resume and retry failed questions.")
    return 1 if failures or incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
