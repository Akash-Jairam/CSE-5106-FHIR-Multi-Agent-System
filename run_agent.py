import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from importlib import metadata
from pathlib import Path

import pandas as pd
from packaging.requirements import Requirement
from tqdm import tqdm

from utils import (
    RESULT_COLUMNS,
    check_tool_credentials,
    get_default_model,
    curate_input_dataset,
    read_input_data,
    run_agent_with_input,
    validate_config,
)

INPUT_COLUMNS = ["question_id", "question", "true_answer", "assumption", "patient_fhir_id"]
HISTORY_COLUMN = "attempt_history"
CONDITIONS = ("intact", "omitted")


def dispatch_agent_job(args_tuple):
    """Additive V3 dispatch through the common budget and audit job contract."""
    if args_tuple[1] != "multi_agent_v3":
        return run_agent_with_input(args_tuple)
    from agent.query_contract_agent import run_v3_job
    return run_v3_job(args_tuple)


def _boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate FHIR agent answers and save results.")
    parser.add_argument("--model", default=get_default_model(),
                        help="Base model; default: $MODEL, then MODEL in config.yml, then gpt-6-luna")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--input", default="final_dataset/questions_answers_sql_fhir.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--date", default=None, help="Retained for compatibility")
    parser.add_argument("--agent_strategy", required=True, choices=[
        "single_turn_request", "single_turn_resource", "single_turn_code_resource",
        "multi_turn_resource", "multi_turn_code_resource", "multi_turn_code_resource_v2",
        "direct_llm", "single_agent", "multi_agent", "multi_agent_v3",
    ])
    parser.add_argument("--add_patient_fhir_id", type=_boolean, nargs="?", const=True, default=True)
    parser.add_argument("--num_processes", type=int, default=-1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_false", dest="verbose")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split_manifest", default=None, help="Frozen patient-disjoint split JSON")
    parser.add_argument("--split", choices=["pilot", "dev", "held_out"], default=None)
    parser.add_argument("--max_total_tokens", type=int, default=None)
    parser.add_argument("--max_tool_calls", type=int, default=None)
    parser.add_argument("--enable_cache", type=_boolean, nargs="?", const=True, default=True)
    parser.add_argument("--audit_storage", choices=("inline", "gzip"), default="inline",
                        help="gzip preserves full trace/resources in lossless artifacts before rows accumulate; default inline")
    parser.add_argument("--perturbations", default=None,
                        help="Frozen missing-evidence manifest; restricts the run to its questions")
    parser.add_argument("--condition", choices=CONDITIONS, default=None,
                        help="With --perturbations: start from the full (intact) or the omitted packet")
    parser.add_argument("--subset", default=None,
                        help="Frozen question subset (manifests/variability_subset.json); restricts the run to it")
    parser.add_argument("--repeat", type=int, default=None,
                        help="Repeat number of a repeated subset run; recorded in the run identity only")
    return parser


def prepare_multiprocessing_inputs(remaining_inputs, agent_strategy, model, verbose, base_url, enable_cache, budget,
                                   perturbations=None, include_full_audit=False, audit_contexts=None):
    perturbations = perturbations or [None] * len(remaining_inputs)
    jobs = [(input_data, agent_strategy, model, verbose, base_url, enable_cache, budget, perturbation)
            for input_data, perturbation in zip(remaining_inputs, perturbations)]
    if audit_contexts is not None and len(audit_contexts) != len(jobs):
        raise ValueError("Every audit job needs its owning question/output context")
    if audit_contexts is not None:
        return [(*job, True, context) for job, context in zip(jobs, audit_contexts)]
    return [(*job, True) for job in jobs] if include_full_audit else jobs


def load_subset(path, split_manifest, split):
    """Question IDs of a frozen subset, after checking it was drawn from this split manifest."""
    with open(path, encoding="utf-8") as stream:
        subset = json.load(stream)
    question_ids = subset.get("question_ids")
    if subset.get("version") != 1 or not isinstance(question_ids, list) or len(question_ids) != subset.get("size"):
        raise ValueError("The subset manifest is malformed; rebuild it")
    if subset.get("split") != split:
        raise ValueError(f"The subset was drawn from the {subset.get('split')} split, not {split}")
    if hashlib.sha256(Path(split_manifest).read_bytes()).hexdigest() != subset.get("split_sha256"):
        raise ValueError("The subset was drawn from a different split manifest; rebuild it")
    return question_ids


def load_inputs(input_file, limit=None, split_manifest=None, split=None, question_ids=None):
    """The selected split, then only `question_ids` (in split order), then the first `limit`."""
    df = read_input_data(input_file)
    if split_manifest:
        with open(split_manifest, encoding="utf-8") as stream:
            manifest = json.load(stream)
        source_hash = manifest.get("source_sha256")
        if source_hash and hashlib.sha256(Path(input_file).read_bytes()).hexdigest() != source_hash:
            raise ValueError("The split manifest was built from a different input dataset")
        df = select_split(df, manifest, split)
    if question_ids is not None:
        missing = set(question_ids) - set(df["question_id"])
        if missing:
            raise ValueError(f"{len(missing)} selected questions are not in the selected inputs")
        df = df[df["question_id"].isin(question_ids)].reset_index(drop=True)
    return df.head(limit) if limit is not None else df


def select_split(df, manifest, selected):
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or selected not in splits:
        raise ValueError("The requested split is missing from the split manifest")
    if df["question_id"].isna().any() or df["question_id"].duplicated().any():
        raise ValueError("Each input question must have a unique, non-null question_id")
    indexed = df.set_index("question_id", drop=False)
    seen_patients, seen_questions = set(), set()
    for name, entry in splits.items():
        question_ids = entry.get("question_ids", [])
        patients = [patient["patient_fhir_id"] for patient in entry.get("patients", [])]
        if len(set(question_ids)) != len(question_ids) or len(set(patients)) != len(patients):
            raise ValueError(f"Split {name} has duplicate question or patient IDs")
        if seen_patients.intersection(patients) or seen_questions.intersection(question_ids):
            raise ValueError("The split manifest is not patient-disjoint")
        if not set(question_ids) <= set(indexed.index):
            raise ValueError(f"Split {name} refers to questions absent from the input dataset")
        if not set(indexed.loc[question_ids, "patient_fhir_id"]) <= set(patients):
            raise ValueError(f"Split {name} has a question assigned to the wrong patient pool")
        for count_key in ("family_counts", "answer_kind_counts"):
            counts = entry.get(count_key)
            if counts is not None and sum(counts.values()) != len(question_ids):
                raise ValueError(f"Split {name} {count_key} does not match its question count")
        seen_patients.update(patients)
        seen_questions.update(question_ids)
    return indexed.loc[splits[selected]["question_ids"]].reset_index(drop=True)


def code_fingerprint():
    root = Path(__file__).resolve().parent
    paths = [root / "run_agent.py", root / "fhir_client.py", root / "requirements.txt"]
    for filename in ("audit_store.py", "full_benchmark.py"):
        helper = root / "scripts" / filename
        if helper.exists():
            paths.append(helper)
    for directory in ("agent", "tools", "utils"):
        paths.extend((root / directory).rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(json.dumps(runtime_dependency_versions(root / "requirements.txt"), sort_keys=True).encode())
    return digest.hexdigest()


def runtime_dependency_versions(requirements_path):
    """Public package versions, never model settings or credential files."""
    names = set()
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirement = Requirement(line)
            if requirement.marker is None or requirement.marker.evaluate():
                names.add(requirement.name)
    if "langgraph" in names:
        # These transitive packages implement graph execution and local tracing.
        names.update(("langchain-core", "langgraph-checkpoint", "langgraph-prebuilt", "langgraph-sdk", "langsmith"))
    versions = {}
    for name in sorted(names):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _present(value):
    return value is not None and not (isinstance(value, float) and pd.isna(value))


def is_done(row):
    """A successful abstention is complete; a failed attempt can be retried on resume."""
    if _present(row.get("error")) and row.get("error") != "":
        return False
    return _present(row.get("agent_answer")) or _present(row.get("latency"))


def is_attempted(row):
    """A terminal execution includes errors; an unscheduled placeholder does not."""
    return is_done(row) or (_present(row.get("error")) and row.get("error") != "")


def load_saved_results(output_file, inputs):
    """Validate scheduled identities before indexing, resuming, or pairing results."""
    if not os.path.exists(output_file):
        return {}
    with open(output_file, encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError("Saved results must be a list of question records")
    patients = dict(zip(inputs["question_id"], inputs["patient_fhir_id"]))
    rows = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Saved results must contain question records")
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid or qid in rows:
            raise ValueError("Saved results must contain unique, non-null question_id values")
        if qid not in patients:
            raise ValueError("Saved results contain a question outside the scheduled inputs")
        if row.get("patient_fhir_id") != patients[qid]:
            raise ValueError("Saved results contain a question assigned to the wrong patient")
        history = row.get(HISTORY_COLUMN)
        if history is not None and (not isinstance(history, list)
                                    or any(not isinstance(attempt, dict) or not is_attempted(attempt)
                                           for attempt in history)):
            raise ValueError("Attempt history must be a list of recorded result objects")
        rows[qid] = row
    return rows


def run_identity(args, inputs):
    """Bind a result file to the experiment and inputs that produced it."""
    input_json = inputs[INPUT_COLUMNS].to_json(orient="records", date_format="iso")
    perturbations = getattr(args, "perturbations", None)
    subset = getattr(args, "subset", None)
    return {
        "version": 2,
        "agent_strategy": args.agent_strategy,
        "model": args.model,
        "base_url_hash": hashlib.sha256((args.base_url or "").encode()).hexdigest(),
        "max_total_tokens": args.max_total_tokens,
        "max_tool_calls": args.max_tool_calls,
        "add_patient_fhir_id": args.add_patient_fhir_id,
        "enable_cache": args.enable_cache,
        "input_hash": hashlib.sha256(input_json.encode()).hexdigest(),
        "code_hash": code_fingerprint(),
        "split": getattr(args, "split", None),
        "split_manifest_hash": (hashlib.sha256(Path(args.split_manifest).read_bytes()).hexdigest()
                                if getattr(args, "split_manifest", None) else None),
        "condition": getattr(args, "condition", None),
        "perturbations_hash": hashlib.sha256(Path(perturbations).read_bytes()).hexdigest() if perturbations else None,
        "subset_hash": hashlib.sha256(Path(subset).read_bytes()).hexdigest() if subset else None,
        "repeat": getattr(args, "repeat", None),
        "audit_storage": getattr(args, "audit_storage", "inline"),
        "audit_artifact_schema": 2 if getattr(args, "audit_storage", "inline") == "gzip" else None,
    }


def _write_json(value, output_file):
    temporary = output_file + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(_json_values(value), stream, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        if isinstance(value, list):
            from scripts.full_benchmark import iter_result_rows
            validated = sum(1 for _ in iter_result_rows(temporary))
            if validated != len(value):
                raise ValueError("Serialized result count changed during validation")
        else:
            with open(temporary, encoding="utf-8") as stream:
                json.load(stream)
        os.replace(temporary, output_file)
    except Exception:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def _json_values(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    return value


def save_results_json(df, output_file):
    # Object conversion makes unfinished rows and empty reference answers JSON null.
    records = df.astype(object).where(df.notna(), None).to_dict("records")
    _write_json(records, output_file)
    print(f"Saved {sum(is_done(row) for row in records)} completed questions to {output_file}")


def prepare_results(inputs, output_file, identity):
    if inputs["question_id"].isna().any() or inputs["question_id"].duplicated().any():
        raise ValueError("Each input question must have a unique, non-null question_id")
    metadata_file = output_file + ".run.json"
    if os.path.exists(output_file):
        if not os.path.exists(metadata_file):
            raise ValueError("Existing results have no run identity; use a new output path to avoid mixing experiments")
        with open(metadata_file, encoding="utf-8") as stream:
            saved_identity = json.load(stream)
        if saved_identity != identity:
            raise ValueError("Existing results belong to different inputs or run settings; use a new output path")
    saved = load_saved_results(output_file, inputs)
    if identity.get("audit_storage") == "gzip":
        from scripts.audit_store import verify_result_audit
        for question_id, row in saved.items():
            if is_attempted(row):
                if row.get("audit_artifacts") is None:
                    raise ValueError("A gzip attempt is missing its complete audit reference")
                verify_result_audit(row, Path(output_file).resolve().parent, question_id=question_id)
    previous = pd.DataFrame(saved.values())
    if not previous.empty:
        columns = ["question_id"] + RESULT_COLUMNS + [HISTORY_COLUMN]
        previous = previous.reindex(columns=columns)
        results = inputs.merge(previous, on="question_id", how="left", validate="one_to_one")
    else:
        results = inputs.copy()
        for column in RESULT_COLUMNS + [HISTORY_COLUMN]:
            results[column] = None
    results[RESULT_COLUMNS + [HISTORY_COLUMN]] = results[RESULT_COLUMNS + [HISTORY_COLUMN]].astype(object)
    return results


def store_attempt(results, index, output, audit_storage="inline", output_file=None):
    if audit_storage == "gzip":
        if output_file is None:
            raise ValueError("gzip audit storage requires the owning output file")
        from scripts.audit_store import externalize_result_audit
        output = externalize_result_audit(output, Path(output_file).resolve().parent,
                                          question_id=results.at[index, "question_id"])
    elif audit_storage != "inline":
        raise ValueError("Unknown audit storage mode")
    previous = results.loc[index]
    if _present(previous.get("latency")) or _present(previous.get("error")):
        history = previous.get(HISTORY_COLUMN)
        if _present(history) and (not isinstance(history, list)
                                  or any(not isinstance(attempt, dict) or not is_attempted(attempt)
                                         for attempt in history)):
            raise ValueError("Attempt history must be a list of recorded result objects")
        history = list(history) if isinstance(history, list) else []
        history.append({column: previous.get(column) for column in RESULT_COLUMNS})
        results.at[index, HISTORY_COLUMN] = history
    for key, value in output.items():
        results.at[index, key] = value


def print_usage_summary(df):
    """Include prior failed attempts and distinguish executed calls from proposed ones."""
    attempts = []
    for row in df.to_dict("records"):
        attempts.extend(row.get(HISTORY_COLUMN) if isinstance(row.get(HISTORY_COLUMN), list) else [])
        attempts.append(row)
    tokens = [row["tokens"] for row in attempts if isinstance(row.get("tokens"), dict)]
    budgets = [row["budget"] for row in attempts if isinstance(row.get("budget"), dict)]
    if not tokens:
        return
    total = lambda key: sum(t.get(key, 0) or 0 for t in tokens)
    costs_known = all(t.get("cost") is not None for t in tokens)
    cost = f"${total('cost'):.4f}" if costs_known else "cost unavailable for one or more attempts"
    print(f"Usage over {len(tokens)} attempts: {total('total'):,} tokens "
          f"({total('prompt'):,} prompt, {total('completion'):,} completion), "
          f"{total('llm_calls')} model calls; {cost}")
    print(f"  Repeated prompt tokens: {total('repeated_prompt'):,}; cached prompt tokens: {total('cached_prompt'):,}")
    print(f"  Tool calls executed: {sum(b.get('tool_calls_run', 0) or 0 for b in budgets)}; "
          f"attempts stopped by a ceiling: {sum(bool(b.get('stopped')) for b in budgets)}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for key in ("max_total_tokens", "max_tool_calls", "limit"):
        value = getattr(args, key)
        if value is not None and value < 0:
            parser.error(f"--{key} must be non-negative")
    if args.save_interval < 1 or args.num_processes == 0 or args.num_processes < -1:
        parser.error("--save_interval must be positive; --num_processes must be positive or -1")
    if bool(args.split_manifest) != bool(args.split):
        parser.error("--split_manifest and --split must be supplied together")
    if bool(args.perturbations) != bool(args.condition):
        parser.error("--perturbations and --condition must be supplied together")
    if args.perturbations and args.split != "held_out":
        parser.error("Perturbations are frozen on held-out questions; use --split held_out")
    if args.subset and (args.perturbations or not args.split):
        parser.error("--subset needs --split_manifest and --split, and cannot be combined with --perturbations")
    if args.repeat is not None and (not args.subset or args.repeat < 1):
        parser.error("--repeat must be a positive repeat number of a --subset run")
    question_ids = load_subset(args.subset, args.split_manifest, args.split) if args.subset else None
    perturbations = None
    if args.perturbations:
        from tools.perturbation import load_perturbations
        perturbations = load_perturbations(args.perturbations)
    output_file = args.output or f"output/{args.agent_strategy}_{args.model.replace('/', '_')}_results.json"
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    if perturbations is not None:
        question_ids = list(perturbations)
    inputs = load_inputs(args.input, args.limit, args.split_manifest, args.split, question_ids=question_ids)
    identity = run_identity(args, inputs)
    results = prepare_results(inputs, output_file, identity)
    results["question_with_context"] = curate_input_dataset(results, args.add_patient_fhir_id)
    remaining = results.loc[~results.apply(is_done, axis=1)].copy()
    print(f"Completed questions: {len(results) - len(remaining)}; remaining: {len(remaining)}")
    if remaining.empty:
        if not os.path.exists(output_file):
            _write_json(identity, output_file + ".run.json")
            save_results_json(results, output_file)
        print_usage_summary(results)
        return 0
    if not validate_config(args.model, args.base_url):
        return 1
    check_tool_credentials()
    _write_json(identity, output_file + ".run.json")
    budget = {"max_total_tokens": args.max_total_tokens, "max_tool_calls": args.max_tool_calls}
    # Intact runs get no specification, so they start from the same full packet as ordinary runs.
    specifications = ([perturbations[qid] for qid in remaining["question_id"]]
                      if args.condition == "omitted" else None)
    jobs = prepare_multiprocessing_inputs(remaining["question_with_context"].tolist(), args.agent_strategy,
                                         args.model, args.verbose, args.base_url, args.enable_cache, budget,
                                         specifications, include_full_audit=args.audit_storage == "gzip",
                                         audit_contexts=([{"output_dir": str(Path(output_file).resolve().parent),
                                                          "question_id": question_id}
                                                         for question_id in remaining["question_id"]]
                                                        if args.audit_storage == "gzip" else None))
    processes = min(args.num_processes if args.num_processes > 0 else (os.cpu_count() or 1), len(jobs))
    indexes = remaining.index.tolist()

    def collect(outputs):
        for i, output in enumerate(tqdm(outputs, total=len(jobs))):
            store_attempt(results, indexes[i], output, audit_storage=args.audit_storage, output_file=output_file)
            if (i + 1) % args.save_interval == 0:
                save_results_json(results, output_file)

    if processes == 1:
        collect(map(dispatch_agent_job, jobs))
    else:
        with mp.Pool(processes=processes) as pool:
            collect(pool.imap(dispatch_agent_job, jobs))
    save_results_json(results, output_file)
    print_usage_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
