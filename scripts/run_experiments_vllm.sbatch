#!/bin/bash
# Phase 3 experiment with a model served by vLLM (default gpt-oss:120b) as a Slurm batch job on RIS Compute2.
# Ollama 0.40.13 rejects gpt-oss replies whose tool-call arguments are not valid JSON with HTTP 500;
# vLLM implements gpt-oss's tool-call format and returns such calls, which the agents report back
# to the model as a tool error, as for any other model.
# Starts vLLM on the job's GPU, runs scripts/run_experiments.py against it on the same node, then stops
# vLLM. Rerunning resumes where the last job stopped.
# Setup and usage: docs/compute2-gpu.md, "Serving gpt-oss with vLLM".
#
# Usage (from the repository root on the login node):
#   sbatch --job-name=phase3-gpt-oss-vllm scripts/run_experiments_vllm.sbatch
#   SPLIT=dev sbatch --job-name=phase3-gpt-oss-vllm scripts/run_experiments_vllm.sbatch --arms ordinary_multi_agent
#SBATCH --job-name=phase3-vllm
#SBATCH --account=compute2-di2cap
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail

model="${MODEL:-gpt-oss:120b}"            # the name the runner and results use
hf_model="${HF_MODEL:-openai/gpt-oss-120b}"
split="${SPLIT:-held_out}"
scratch="/scratch2/fs1/di2cap/$USER"
vllm_venv="${VLLM_VENV:-$scratch/vllm-venv}"
# Weights are downloaded once on the login node; the job reads them offline
export HF_HOME="${HF_HOME:-$scratch/hf}" HF_HUB_OFFLINE=1
# vLLM compiles FlashInfer kernels at startup with the venv's ninja and the node's nvcc; keep its caches off home
export PATH="$vllm_venv/bin:$PATH" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" XDG_CACHE_HOME="$scratch/cache"
workers="${WORKERS:-4}"
# Not 8000: other users' servers on a shared node often hold it. Keep the port fixed across jobs,
# because result files are bound to the base URL
port=28431
base_url="http://127.0.0.1:${port}/v1"
if curl -s --max-time 5 "http://127.0.0.1:${port}/" > /dev/null; then
  echo "Port $port is already in use on $(hostname)"; exit 1
fi

# Loopback only and the same 32k context as the Ollama runs
"$vllm_venv/bin/vllm" serve "$hf_model" --served-model-name "$model" --host 127.0.0.1 \
  --port "$port" --max-model-len 32768 --max-num-seqs "$workers" \
  --tool-call-parser openai --enable-auto-tool-choice \
  > "logs/vllm-${SLURM_JOB_ID}.log" 2>&1 &
vllm_pid=$!
trap 'kill "$vllm_pid" 2>/dev/null' EXIT

# Loading 65 GB of weights and compiling takes several minutes. The reply must name our model,
# so another server on the same port is never mistaken for it
ready() { curl -sf --max-time 10 "http://127.0.0.1:${port}/v1/models" | grep -q "\"$model\""; }
for _ in $(seq 360); do
  ready && break
  kill -0 "$vllm_pid" 2>/dev/null || break
  sleep 5
done
if ! ready; then
  echo "vLLM did not start; see logs/vllm-${SLURM_JOB_ID}.log"
  curl -sS -v --max-time 10 "http://127.0.0.1:${port}/v1/models" 2>&1 | tail -20
  ss -ltnp 2>/dev/null | grep ":${port} "
  exit 1
fi

echo "Node $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader), $model via vLLM, $split split, $workers workers"
source .venv/bin/activate

# The agents' answers depend on the Python sandbox; stop now rather than after hours of runs without it
python -c 'from tools.python_executor import execute_python_code as run; r = run("answer = 1 + 1"); assert r == {"answer": 2}, r' \
  || { echo "Python sandbox does not work on this node"; exit 1; }

common=(--model "$model" --base_url "$base_url" --max_total_tokens 200000 --max_tool_calls 30 --split "$split")
# Exits non-zero while any question has failed; the status below says whether to resubmit
python scripts/run_experiments.py run "${common[@]}" --num_processes "$workers" "$@"
python scripts/run_experiments.py status "${common[@]}" "$@"
