#!/bin/bash
# Submit the new question suites (docs/question_generation.md) for one model on RIS Compute2.
# Prints the sbatch commands; set SUBMIT=1 to submit them. A model's jobs share one job name with
# --dependency=singleton, so they run one at a time (the vLLM job serves on a fixed port), in order.
# Rerunning a submitted suite resumes it.
#
# Usage (from the repository root on the login node):
#   scripts/submit_question_suites.sh gpt-oss:120b            # print
#   SUBMIT=1 scripts/submit_question_suites.sh qwen3:32b      # submit
#   SUITES="pool_held_out_v2 paraphrases" scripts/submit_question_suites.sh gpt-oss:120b
set -euo pipefail

model="${1:?usage: $0 MODEL (gpt-oss:120b or qwen3:32b)}"
case "$model" in
  gpt-oss:*) script=scripts/run_experiments_vllm.sbatch ;;
  *) script=scripts/run_experiments_qwen.sbatch ;;
esac
out="output/experiments/$model"
generated=(--input final_dataset/generated/fhir_native.csv --families manifests/generated/fhir_native.families.csv)
paraphrases=(--input final_dataset/generated/paraphrases.csv --families manifests/generated/paraphrases.families.csv)

suite_args() {
  case "$1" in
    pool_*) echo --arms ordinary --split_manifest "manifests/pools/${1#pool_}.json" ;;
    paraphrases) echo --arms ordinary "${paraphrases[@]}" --split_manifest manifests/generated/paraphrases.split.json ;;
    generated) echo --arms ordinary "${generated[@]}" --split_manifest manifests/generated/fhir_native.split.json ;;
    generated_positive) echo --arms ordinary "${generated[@]}" \
      --split_manifest manifests/generated/fhir_native.positive.split.json ;;
    multi_perturbations) echo --arms recovery --perturbations manifests/perturbations_multi.json ;;
    *) echo "unknown suite $1" >&2; return 1 ;;
  esac
}

default="pool_held_out_v2 pool_template_balanced pool_zero_finisher pool_truncation pool_disagreement pool_out_of_scope
paraphrases generated generated_positive multi_perturbations"
name="suites-${model//[:\/]/-}"
mkdir -p logs
for suite in ${SUITES:-$default}; do
  args=$(suite_args "$suite")
  # shellcheck disable=SC2086
  command=(sbatch --job-name="$name" --dependency=singleton "$script" $args --output_dir "$out/suite_$suite")
  if [ "${SUBMIT:-0}" = 1 ]; then
    "${command[@]}"
  else
    echo "${command[*]}"
  fi
done
