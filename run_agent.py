import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils import (
    RESULT_COLUMNS,
    check_tool_credentials,
    curate_input_dataset,
    read_input_data,
    read_intermediate_results,
    run_agent_with_input,
    validate_config,
)

INPUT_COLUMNS = ["question_id", "question", "true_answer", "assumption", "patient_fhir_id"]
HISTORY_COLUMN = "attempt_history"


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
    parser.add_argument("--model", default="o4-mini")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--input", default="final_dataset/questions_answers_sql_fhir.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--date", default=None, help="Retained for compatibility")
    parser.add_argument("--agent_strategy", required=True, choices=[
        "single_turn_request", "single_turn_resource", "single_turn_code_resource",
        "multi_turn_resource", "multi_turn_code_resource", "multi_turn_code_resource_v2",
        "direct_llm", "single_agent",
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
    return parser


def prepare_multiprocessing_inputs(remaining_inputs, agent_strategy, model, verbose, base_url, enable_cache, budget):
    return [(input_data, agent_strategy, model, verbose, base_url, enable_cache, budget)
            for input_data in remaining_inputs]


def load_inputs(input_file, limit=None, split_manifest=None, split=None):
    df = read_input_data(input_file)
    if split_manifest:
        with open(split_manifest, encoding="utf-8") as stream:
            manifest = json.load(stream)
        source_hash = manifest.get("source_sha256")
        if source_hash and hashlib.sha256(Path(input_file).read_bytes()).hexdigest() != source_hash:
            raise ValueError("The split manifest was built from a different input dataset")
        df = select_split(df, manifest, split)
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
    paths = [root / "run_agent.py", root / "fhir_client.py"]
    for directory in ("agent", "tools", "utils"):
        paths.extend((root / directory).rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _present(value):
    return value is not None and not (isinstance(value, float) and pd.isna(value))


def is_done(row):
    """A successful abstention is complete; a failed attempt can be retried on resume."""
    if _present(row.get("error")) and row.get("error") != "":
        return False
    return _present(row.get("agent_answer")) or _present(row.get("latency"))


def run_identity(args, inputs):
    """Bind a result file to the experiment and inputs that produced it."""
    input_json = inputs[INPUT_COLUMNS].to_json(orient="records", date_format="iso")
    return {
        "version": 1,
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
    }


def _write_json(value, output_file):
    temporary = output_file + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(_json_values(value), stream, ensure_ascii=False, indent=2, default=str, allow_nan=False)
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
    previous = read_intermediate_results(output_file)
    metadata_file = output_file + ".run.json"
    if not previous.empty:
        if not os.path.exists(metadata_file):
            raise ValueError("Existing results have no run identity; use a new output path to avoid mixing experiments")
        with open(metadata_file, encoding="utf-8") as stream:
            saved_identity = json.load(stream)
        if saved_identity != identity:
            raise ValueError("Existing results belong to different inputs or run settings; use a new output path")
        if "question_id" not in previous or previous["question_id"].duplicated().any():
            raise ValueError("Saved results must contain unique question_id values")
        columns = ["question_id"] + RESULT_COLUMNS + [HISTORY_COLUMN]
        previous = previous.reindex(columns=columns)
        results = inputs.merge(previous, on="question_id", how="left", validate="one_to_one")
    else:
        results = inputs.copy()
        for column in RESULT_COLUMNS + [HISTORY_COLUMN]:
            results[column] = None
    results[RESULT_COLUMNS + [HISTORY_COLUMN]] = results[RESULT_COLUMNS + [HISTORY_COLUMN]].astype(object)
    return results


def store_attempt(results, index, output):
    previous = results.loc[index]
    if _present(previous.get("latency")) or _present(previous.get("error")):
        history = previous.get(HISTORY_COLUMN)
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
    output_file = args.output or f"output/{args.agent_strategy}_{args.model.replace('/', '_')}_results.json"
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    inputs = load_inputs(args.input, args.limit, args.split_manifest, args.split)
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
    jobs = prepare_multiprocessing_inputs(remaining["question_with_context"].tolist(), args.agent_strategy,
                                         args.model, args.verbose, args.base_url, args.enable_cache, budget)
    processes = min(args.num_processes if args.num_processes > 0 else (os.cpu_count() or 1), len(jobs))
    indexes = remaining.index.tolist()

    def collect(outputs):
        for i, output in enumerate(tqdm(outputs, total=len(jobs))):
            store_attempt(results, indexes[i], output)
            if (i + 1) % args.save_interval == 0:
                save_results_json(results, output_file)

    if processes == 1:
        collect(map(run_agent_with_input, jobs))
    else:
        with mp.Pool(processes=processes) as pool:
            collect(pool.imap(run_agent_with_input, jobs))
    save_results_json(results, output_file)
    print_usage_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
