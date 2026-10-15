"""Validate frozen exposure strata for completed full-run offline analysis only.

No answer, question text, SQL, model, credential or inference code is inspected
here. The operational completion/integrity guard must pass before comparison.
"""
import hashlib
import json
from pathlib import Path
import re

PRIMARY = ("development_patients", "pilot_patients", "adaptive_questions", "question_confirmation")
VIEWS = ("confirmation_in_scope", "confirmation_other", "nominal_new_patient")
UPSTREAM = ("manifests/split.json", "manifests/development_exposure.json")
HEX = re.compile(r"^[a-f0-9]{64}$")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key in evaluation metadata")
        result[key] = value
    return result


def read_json(path):
    def invalid_constant(value):
        raise ValueError("Non-finite JSON in evaluation metadata")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                      parse_constant=invalid_constant)


def question_ids(entry, name, source):
    require(isinstance(entry, dict), name + ": descriptor must be an object")
    ids = entry.get("question_ids")
    require(isinstance(ids, list) and ids and
            all(isinstance(q, str) and q.strip() for q in ids), name + ": nonempty question IDs required")
    require(len(ids) == len(set(ids)), name + ": duplicate question IDs")
    require(set(ids) <= set(source), name + ": unknown question IDs")
    for field, expected in (("questions", len(ids)),
                            ("patients", len({source[q]["patient_fhir_id"] for q in ids}))):
        if field in entry:
            require(type(entry[field]) is int and entry[field] == expected, name + ": declared " + field + " differs")
    return set(ids)


def load_evaluation_strata(path, expected_sha256, root, source, families, input_path, families_path):
    """Bind exact IDs and exposure semantics using metadata, never answer values."""
    require(isinstance(expected_sha256, str) and HEX.fullmatch(expected_sha256), "Frozen evaluation-strata SHA256 required")
    path, root = Path(path).resolve(), Path(root).resolve()
    require(path.suffix == ".json", "Evaluation strata must be a JSON manifest")
    require(sha256(path) == expected_sha256, "Frozen evaluation-strata manifest digest differs")
    manifest = read_json(path)
    require(isinstance(manifest, dict) and type(manifest.get("version")) is int and manifest["version"] == 1,
            "Unsupported evaluation-strata manifest version")
    require(manifest.get("dataset_sha256") == sha256(input_path), "Evaluation dataset digest differs")
    require(manifest.get("families_sha256") == sha256(families_path), "Evaluation families digest differs")
    require(set(source) == set(families), "Full source/family question IDs differ")
    require(all(isinstance(r.get("family"), str) and r["family"].strip() for r in families.values()),
            "Full family metadata needs nonempty labels")
    require(all(isinstance(r.get("patient_fhir_id"), str) and r["patient_fhir_id"].strip()
                for r in source.values()), "Full source needs patient IDs")
    upstream = manifest.get("upstream_sha256")
    require(isinstance(upstream, dict) and set(upstream) == set(UPSTREAM), "Exactly the canonical split/exposure dependencies are required")
    for relative in UPSTREAM:
        require(isinstance(upstream[relative], str) and HEX.fullmatch(upstream[relative])
                and sha256(root / relative) == upstream[relative], "Frozen evaluation dependency differs: " + relative)
    for field, relative in (("split_manifest_sha256", UPSTREAM[0]), ("development_exposure_sha256", UPSTREAM[1])):
        require(manifest.get(field) == upstream[relative], "Evaluation dependency aliases differ")
    split, exposure = (read_json(root / relative) for relative in UPSTREAM)
    old = split.get("splits") or {}
    require(all(isinstance(old.get(k), dict) for k in ("dev", "pilot", "held_out")), "Canonical split metadata is incomplete")
    dev = {p["patient_fhir_id"] for p in old["dev"]["patients"]}
    pilot = {p["patient_fhir_id"] for p in old["pilot"]["patients"]}
    adaptive = set(old["held_out"]["question_ids"])
    require(dev and pilot and not dev & pilot and adaptive <= set(source), "Canonical exposure metadata is inconsistent")
    require(not {source[q]["patient_fhir_id"] for q in adaptive} & (dev | pilot), "Adaptive questions overlap reserved patients")
    reserved_subjects = {p["subject_id"] for p in old["dev"]["patients"]}
    exposed_subjects = {int(p) for item in exposure.get("exposures", []) for p in item["subject_ids"]}
    require(exposed_subjects <= reserved_subjects, "Documented exposed subjects escaped development")
    expected = {"development_patients": {q for q, r in source.items() if r["patient_fhir_id"] in dev},
                "pilot_patients": {q for q, r in source.items() if r["patient_fhir_id"] in pilot},
                "adaptive_questions": adaptive}
    expected["question_confirmation"] = set(source) - set.union(*expected.values())
    splits = manifest.get("splits")
    require(isinstance(splits, dict) and set(splits) == set(PRIMARY), "Exactly four primary exposure strata are required")
    groups, seen = {}, set()
    for name in PRIMARY:
        ids = question_ids(splits[name], name, source)
        require(not seen & ids, "Primary evaluation strata overlap")
        require(ids == expected[name], name + ": frozen patient/question exposure membership differs")
        seen |= ids
        groups[name] = sorted(ids)
    require(seen == set(source), "Primary evaluation strata do not cover the full cohort")
    if "full_question_ids" in manifest:
        require(set(manifest["full_question_ids"]) == set(source) and
                len(manifest["full_question_ids"]) == len(source), "Declared full question IDs differ")
    views = manifest.get("views", {})
    require(isinstance(views, dict) and set(views) <= set(VIEWS), "Unknown evaluation views")
    checked_views = {name: question_ids(entry, name, source) for name, entry in views.items()}
    confirmation = expected["question_confirmation"]
    require(all(ids <= confirmation for ids in checked_views.values()), "Views must be nested inside question confirmation")
    if set(checked_views) & {"confirmation_in_scope", "confirmation_other"}:
        require({"confirmation_in_scope", "confirmation_other"} <= set(checked_views), "Both confirmation scope views are required together")
        in_scope = set(split.get("in_scope_families") or [])
        require(in_scope, "Canonical in-scope family metadata is missing")
        expected_in = {q for q in confirmation if families[q]["family"] in in_scope}
        require(checked_views["confirmation_in_scope"] == expected_in and
                checked_views["confirmation_other"] == confirmation - expected_in,
                "Confirmation scope views differ from family metadata")
    if "nominal_new_patient" in checked_views:
        used = dev | pilot | {source[q]["patient_fhir_id"] for q in adaptive}
        nominal = {q for q, r in source.items() if r["patient_fhir_id"] not in used}
        require(checked_views["nominal_new_patient"] == nominal and
                len({source[q]["patient_fhir_id"] for q in nominal}) == 1,
                "Nominal new-patient view differs from the single metadata candidate")
    require(sha256(path) == expected_sha256, "Evaluation-strata manifest changed while validating")
    return {"manifest_path": str(path), "manifest_sha256": expected_sha256,
            "dataset_sha256": manifest["dataset_sha256"], "families_sha256": manifest["families_sha256"],
            "upstream_sha256": upstream, "splits": groups,
            "views": {name: sorted(ids) for name, ids in checked_views.items()},
            "limits": ["Confirmation questions mostly share patients with development; this is not patient-disjoint validation.",
                       "One-patient views have no patient-population bootstrap interval.",
                       "All stratum denominators include terminal failures and the recorded first attempt.",
                       "Only question_confirmation is primary; other strata/views are descriptive.",
                       "The operational full-terminal and source/audit integrity guard is a prerequisite; this metadata validator does not supervise processes."]}
