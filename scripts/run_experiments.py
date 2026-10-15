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
    python scripts/run_experiments.py plan
    python scripts/run_experiments.py run    --max_total_tokens 200000 --max_tool_calls 30
    python scripts/run_experiments.py status --max_total_tokens 200000 --max_tool_calls 30

    python scripts/run_experiments.py variability --max_total_tokens 200000 --max_tool_calls 30

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
from utils.model_config import get_default_model  # noqa: E402

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
    parser.add_argument("command", choices=["plan", "run", "status", "variability", "analyze", "revisions"])
    parser.add_argument("--model", default=get_default_model(),
                        help="One base model shared by every system; default: $MODEL, then MODEL in config.yml, "
                             "then gpt-6-luna")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--team_strategy", choices=["multi_agent", "multi_agent_v3", "multi_agent_v8"], default="multi_agent",
                        help="Team implementation; arm names stay stable while run identity records the actual strategy")
    parser.add_argument("--verifier_model", default=None,
                        help="multi_agent_v8 only: run the team's program audit on this model (heterogeneous verifier arm)")
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
    parser.add_argument("--audit_storage", choices=("inline", "gzip"), default="inline",
                        help="gzip preserves full audits in lossless artifacts before worker IPC and DataFrame storage")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--split_manifest", default=DEFAULT_SPLIT)
    parser.add_argument("--perturbations", default=DEFAULT_PERTURBATIONS)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--families", default=DEFAULT_FAMILIES, help="Templates for the typed scorer")
    parser.add_argument("--verbose", action="store_true", help="Print each agent's progress")
    parser.add_argument("--bootstrap", type=int, default=10_000, help="Patient-clustered bootstrap resamples")
    parser.add_argument("--evidence_source", choices=["none", "store", "ndjson"], default="none",
                        help="revisions: check released citations against the FHIR store or local NDJSON")
    parser.add_argument("--fhir_dir", default=None, help="Local FHIR NDJSON directory for --evidence_source ndjson")
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
    strategy = getattr(args, "team_strategy", "multi_agent") if arm.strategy == "multi_agent" else arm.strategy
    argv = ["--agent_strategy", strategy, "--model", args.model,
            "--input", args.input, "--output", os.path.join(output_dir(args), f"{arm.name}.json"),
            "--split_manifest", args.split_manifest, "--split", args.split,
            "--enable_cache", "true", "--num_processes", str(args.num_processes),
            "--save_interval", str(args.save_interval), "--audit_storage", getattr(args, "audit_storage", "inline")]
    for key in ("base_url", "max_total_tokens", "max_tool_calls", "limit"):
        if getattr(args, key) is not None:
            argv += [f"--{key}", str(getattr(args, key))]
    if arm.condition:
        argv += ["--perturbations", args.perturbations, "--condition", arm.condition]
    if strategy == "multi_agent_v8" and getattr(args, "verifier_model", None):
        argv += ["--verifier_model", args.verifier_model]
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
              "completed": 0, "failed": 0, "attempted": 0, "analyzable": False, "state": "not started"}
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
    saved_rows = run_agent.load_saved_results(settings.output, inputs)
    rows = {qid: saved_rows.get(qid, {}) for qid in inputs["question_id"]}
    status["completed"] = sum(run_agent.is_done(row) for row in rows.values())
    status["failed"] = sum(not run_agent.is_done(row) and bool(row.get("error")) for row in rows.values())
    status["attempted"] = sum(run_agent.is_attempted(row) for row in rows.values())
    status["analyzable"] = bool(len(inputs)) and status["attempted"] == len(inputs)
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


def load_references(args, question_ids):
    """{question_id: question, reference answer, template, family, patient} for scoring; never printed."""
    import pandas as pd
    from scripts.build_split import file_sha256

    frame = run_agent.read_input_data(args.input)
    frame = frame[frame["question_id"].isin(question_ids)]
    with open(args.split_manifest, encoding="utf-8") as stream:
        expected_hash = json.load(stream).get("family_manifest_sha256")
    if expected_hash and file_sha256(args.families) != expected_hash:
        raise ValueError("Analysis needs the family manifest frozen with the split")
    families = pd.read_csv(args.families, usecols=["question_id", "family", "template"])
    if families["question_id"].isna().any() or families["question_id"].duplicated().any():
        raise ValueError("Analysis families must have unique, non-null question IDs")
    if not set(question_ids) <= set(families["question_id"]):
        raise ValueError("Analysis families are missing scheduled questions")
    families = families.set_index("question_id")
    return {row.question_id: {"question": row.question, "true_answer": row.true_answer,
                              "template": families["template"].get(row.question_id, ""),
                              "family": families["family"].get(row.question_id),
                              "patient": row.patient_fhir_id}
            for row in frame.itertuples()}


def revision_tracking(arms, args, question_sets):
    """(report, per-question records) for the complete arms; see scripts/revisions.py."""
    from scripts.revisions import question_record, summarize_arm, summarize_recovery
    from scripts.variability import first_attempt
    from utils.evidence_scorer import cited_refs, fetch_from_ndjson, fetch_from_store

    results = {arm: arm_results(arm, args, question_sets) for arm in arms}
    complete = {arm: rows for arm, (status, rows) in results.items()
                if status.get("analyzable", status["state"] == "complete")}
    references = load_references(args, {qid for rows in complete.values() for qid in rows})
    specifications = {}
    if any(arm.condition == "omitted" for arm in complete):
        from tools.perturbation import load_perturbations
        specifications = load_perturbations(args.perturbations)
    resources = None
    if args.evidence_source != "none":
        observations = [first_attempt(row) for rows in complete.values() for row in rows.values()]
        refs = cited_refs(row["evidence"] for row in observations if isinstance(row.get("evidence"), list))
        resources = (fetch_from_ndjson(refs, args.fhir_dir) if args.evidence_source == "ndjson"
                     else fetch_from_store(refs))

    records = {arm: [question_record(arm.name, rows[qid], {**references[qid], "question_id": qid},
                                     specifications.get(qid) if arm.condition == "omitted" else None, resources)
                     for qid in sorted(rows)]
               for arm, rows in complete.items()}
    by_name = {arm.name: recs for arm, recs in records.items()}
    recovery = {}
    for strategy in RECOVERY_SYSTEMS:
        omitted = by_name.get(f"recovery_{strategy}_omitted")
        if omitted is not None:
            recovery[strategy] = summarize_recovery(omitted, by_name.get(f"recovery_{strategy}_intact"))
    report = {
        "settings": {"model": args.model, "split": args.split, "limit": args.limit,
                     "evidence_source": args.evidence_source},
        "arms": {arm.name: summarize_arm(recs, resources is not None) for arm, recs in records.items()},
        "recovery": recovery,
        "skipped": {arm.name: status["state"] for arm, (status, _) in results.items() if arm not in complete},
    }
    return report, [record for recs in records.values() for record in recs]


def variability_report(arms, args, question_sets):
    """Per-system spread over all-attempted, current repeats, including errors."""
    from scripts.variability import summarize

    references = load_references(args, question_sets["variability"])
    report = {"subset": os.path.relpath(args.subset, ROOT), "planned_repeats": REPEATS, "systems": {}}
    for strategy in SYSTEMS:
        complete, skipped = [], []
        for arm in (a for a in arms if a.group == "variability" and a.strategy == strategy):
            status, rows = arm_results(arm, args, question_sets)
            (complete if status.get("analyzable", status["state"] == "complete") else skipped).append((arm, rows))
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
        print(f"  error rate        {fmt(s['error_rate'])}")
        print(f"  mean tokens       {fmt(s['mean_tokens'], digits=0)}")
        print(f"  median latency s  {fmt(s['median_latency'], digits=2)}")
        print(f"  retries per repeat {fmt(s['retry_count'], digits=0)}")
        print(f"  retry-inclusive total tokens {fmt(s['retry_inclusive_tokens_total'], digits=0)}")
        print(f"  measured token/latency repeats {s['mean_tokens']['n']}/{s['median_latency']['n']}")
        print(f"  questions always/never/sometimes correct: "
              f"{s['always_correct']}/{s['never_correct']}/{s['unstable']}")
        if s["correctness_agreement"] is not None:
            print(f"  pairwise correctness agreement {s['correctness_agreement']:.3f}; "
                  f"identical answers in every repeat {s['identical_answers']:.3f}")
        if s["skipped_incomplete"]:
            print(f"  skipped incomplete: {', '.join(s['skipped_incomplete'])}")


def print_status(statuses):
    width = max(len(s["arm"]) for s in statuses)
    print(f"{'arm':<{width}}  {'done':>5} {'failed':>6} {'tried':>5} {'total':>5}  state")
    for s in statuses:
        print(f"{s['arm']:<{width}}  {s['completed']:>5} {s['failed']:>6} "
              f"{s['attempted']:>5} {s['questions']:>5}  {s['state']}")
    for group in EXPECTED:
        members = [s for s in statuses if s["group"] == group]
        if members:
            done, total = sum(s["completed"] for s in members), sum(s["questions"] for s in members)
            print(f"{group}: {done}/{total} runs complete")
            print(f"  {sum(s['attempted'] for s in members)}/{total} runs attempted (errors retained for analysis)")


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
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    if (args.evidence_source == "ndjson") != bool(args.fhir_dir):
        parser.error("--fhir_dir is needed with, and only with, --evidence_source ndjson")
    if args.command == "run" and args.split == HELD_OUT:
        if any(getattr(args, key) is None or getattr(args, key) <= 0
               for key in ("max_total_tokens", "max_tool_calls")):
            parser.error("Held-out runs require explicit positive --max_total_tokens and --max_tool_calls ceilings")
    default_arms = {"variability": ["variability"], "analyze": ["ordinary", "recovery"],
                    "revisions": ["ordinary", "recovery"]}
    names = args.arms or default_arms.get(args.command, list(EXPECTED))
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
    if args.command == "revisions":
        from scripts.revisions import print_revisions, write_revisions
        report, records = revision_tracking(arms, args, question_sets)
        directory = os.path.join(output_dir(args), "revisions")
        write_revisions(report, records, directory)
        print_revisions(report)
        print(f"\nWrote {directory}")
        return 0
    if args.command == "analyze":
        from scripts.analysis import analyze, print_analysis, write_analysis
        report = analyze(arms, args, question_sets)
        directory = os.path.join(output_dir(args), "analysis")
        write_analysis(report, directory)
        print_analysis(report)
        print(f"\nWrote {directory}")
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
