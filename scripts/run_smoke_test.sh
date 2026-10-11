# End-to-end smoke test: verify setup, run the single-turn resource agent on a
# few questions, then score the output.
# Usage: MODEL=o4-mini NUM_QUESTIONS=5 bash scripts/run_smoke_test.sh
set -euo pipefail

model="${MODEL:-o4-mini}"
num_questions="${NUM_QUESTIONS:-5}"
agent_strategy="single_turn_resource"
# run_agent.py strips provider prefixes (e.g. "gemini/") from output paths, so do the same here
output="output/smoke_${agent_strategy}_${model##*/}.json"

echo "🔎 Checking environment and FHIR store..."
python scripts/check_setup.py --model "$model"

echo "🤖 Running $agent_strategy agent ($model) on $num_questions questions..."
rm -f "$output"
python run_agent.py \
  --agent_strategy $agent_strategy \
  --model "$model" \
  --input final_dataset/questions_answers_sql_fhir.csv \
  --output "$output" \
  --num_processes 1 \
  --limit "$num_questions"

echo "📊 Evaluating $output..."
python evaluation_metrics.py --input "$output" --model "$model" --num_processes 1 --force
