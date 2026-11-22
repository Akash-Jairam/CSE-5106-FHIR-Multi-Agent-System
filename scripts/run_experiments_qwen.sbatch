#!/bin/bash
# Phase 3 experiment with a local Ollama model (default qwen3:32b) as a Slurm batch job on RIS Compute2.
# Starts Ollama on the job's GPU, pulls the model if it is missing, runs scripts/run_experiments.py
# against it on the same node, then stops Ollama. Rerunning resumes where the last job stopped.
# Setup and usage: docs/compute2-gpu.md, "Batch job: the full Phase 3 experiment".
#
# Usage (from the repository root on the login node):
#   sbatch scripts/run_experiments_qwen.sbatch                        # every group
#   sbatch scripts/run_experiments_qwen.sbatch --arms ordinary        # extra args go to run_experiments.py
#   SPLIT=dev sbatch scripts/run_experiments_qwen.sbatch --arms ordinary_multi_agent
#   MODEL=gpt-oss:120b OLLAMA_MODELS=/scratch2/fs1/di2cap/$USER/ollama-models \
#     sbatch --job-name=phase3-gpt-oss scripts/run_experiments_qwen.sbatch
#SBATCH --job-name=phase3-qwen
#SBATCH --account=compute2-di2cap
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail

model="${MODEL:-qwen3:32b}"
split="${SPLIT:-held_out}"
# Models too large for the 47 GB home quota go on scratch, e.g. gpt-oss:120b (65 GB)
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
workers="${WORKERS:-4}"  # parallel questions; Ollama gets one slot per worker
port=11434
base_url="http://127.0.0.1:${port}/v1"  # keep identical across jobs: result files are bound to it

# Loopback only: nothing outside this node can reach the unauthenticated API
OLLAMA_HOST="127.0.0.1:${port}" OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_NUM_PARALLEL="$workers" \
  "$HOME/ollama/bin/ollama" serve > "logs/ollama-${SLURM_JOB_ID}.log" 2>&1 &
ollama_pid=$!
trap 'kill "$ollama_pid" 2>/dev/null' EXIT

for _ in $(seq 60); do
  curl -sf "http://127.0.0.1:${port}/api/tags" > /dev/null && break
  sleep 2
done
curl -sf "http://127.0.0.1:${port}/api/tags" > /dev/null || { echo "Ollama did not start"; exit 1; }
OLLAMA_HOST="127.0.0.1:${port}" "$HOME/ollama/bin/ollama" pull "$model" > /dev/null \
  || { echo "Could not pull $model into $OLLAMA_MODELS"; exit 1; }

echo "Node $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader), $model, $split split, $workers workers"
source .venv/bin/activate

# The agents' answers depend on the Python sandbox; stop now rather than after hours of runs without it
python -c 'from tools.python_executor import execute_python_code as run; r = run("answer = 1 + 1"); assert r == {"answer": 2}, r' \
  || { echo "Python sandbox does not work on this node"; exit 1; }

common=(--model "$model" --base_url "$base_url" --max_total_tokens 200000 --max_tool_calls 30 --split "$split")
# Exits non-zero while any question has failed; the status below says whether to resubmit
python scripts/run_experiments.py run "${common[@]}" --num_processes "$workers" "$@"
python scripts/run_experiments.py status "${common[@]}" "$@"
