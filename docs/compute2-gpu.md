# Running Qwen on the RIS Compute2 GPU cluster

How to serve a large Qwen model (e.g. `qwen3:32b`) with Ollama on a WashU RIS
Compute2 GPU node and point this project at it from a laptop. The FHIR store
access and API keys stay on the laptop; only the model runs on the cluster.

Last verified 2026-11-02: `qwen3:32b` on one H100 80GB, using about 28 GB with a 32k context.

## At a glance

| Item | Value |
|---|---|
| Login nodes | `c2-login-001.ris.wustl.edu` (also `-002`, `-003`); internal 10.x addresses, so use the WashU network or VPN |
| Username | `akashj` |
| Slurm account | `compute2-di2cap` (**not** `compute2-ris`) |
| Partitions that accept this account | `general-gpu` (H100), and probably `general-preempt-gpu` (H100/H200/A100/B300, can be preempted) |
| Ollama install | `~/ollama` in the shared home (47 GB quota); models in `~/.ollama/models` |
| Laptop endpoint | `BASE_URL=http://localhost:11435/v1` (11434 is the laptop's own Ollama) |

To list your accounts: `sacctmgr -n show assoc user=$USER format=account%40`

## Steps

All interactive steps run in a **separate Terminal window**, not through Claude
Code's `!` prompt. That prompt cannot answer host-key, password, passphrase or
two-factor prompts.

### 1. SSH key (one-time, on the laptop)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519      # passphrase optional
ssh-copy-id -i ~/.ssh/id_ed25519.pub akashj@c2-login-001.ris.wustl.edu
ssh-add --apple-use-keychain ~/.ssh/id_ed25519  # optional: stop passphrase prompts
```

The first `ssh` to a new host asks you to accept its host key; type `yes`.

### 2. Get a GPU (on the login node)

```bash
ssh -i ~/.ssh/id_ed25519 akashj@c2-login-001.ris.wustl.edu
srun --account=compute2-di2cap --partition=general-gpu --gres=gpu:h100:1 \
     --cpus-per-task=8 --mem=64G --time=4:00:00 --pty bash
hostname        # note the node, e.g. c2-gpu-011
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
```

All `srun` options go **before** `--pty bash`; anything after it is passed to bash.

To find free GPUs: `GRES` is the total per node and `GRES_USED` is how many are in use.

```bash
sinfo -N -p general-gpu,general-preempt-gpu -O "nodelist:14,partition:22,gres:22,gresused:32,statecompact:8" | sort -u
```

To check a queued job:

```bash
squeue -u $USER --start -o "%.10i %.20P %.8T %.20S %.30R"
```

`general-interactive` has only two GPU nodes (`c2-gpu-001`/`002`) and is often full.

### 3. Install and start Ollama (on the GPU node)

```bash
mkdir -p ~/ollama
curl -fL https://ollama.com/download/ollama-linux-amd64.tgz | tar -xz -C ~/ollama \
  || curl -fL https://ollama.com/download/ollama-linux-amd64.tar.zst | tar --zstd -x -C ~/ollama

# Listen on all interfaces so the login node can reach it (see step 4)
OLLAMA_HOST=0.0.0.0:11434 OLLAMA_CONTEXT_LENGTH=32768 \
  nohup ~/ollama/bin/ollama serve > ~/ollama/serve.log 2>&1 &
~/ollama/bin/ollama pull qwen3:32b   # about 20 GB; only needed once, home is shared
~/ollama/bin/ollama run qwen3:32b "Say ok" --think=false
```

The install and model download only need to happen once. On later jobs, just
start `ollama serve` again.

### 4. Tunnel from the laptop (separate Terminal, leave it open)

```bash
ssh -i ~/.ssh/id_ed25519 -N -L 11435:c2-gpu-011:11434 akashj@c2-login-001.ris.wustl.edu
```

Replace `c2-gpu-011` with the node from step 2. The window looks frozen while
the tunnel is up.

To check it from the laptop:

```bash
curl -s localhost:11435/api/tags
```

### 5. Run the project against it (on the laptop)

```bash
MODEL=qwen3:32b BASE_URL=http://localhost:11435/v1 NUM_QUESTIONS=3 bash scripts/run_smoke_test.sh
```

### 6. Clean up

On the GPU node, run `pkill -f "ollama serve"`, then `exit` to end the job. Then
close the tunnel window.

## What did not work

- **`ssh -J login akashj@c2-gpu-0NN`** (jumping through the login node into a
  compute node). The node asked for a password and then closed the connection
  with `Connection closed by UNKNOWN port 65535`. Instead, tunnel to the login
  node and forward to `c2-gpu-0NN:11434`, as in step 4.
- **`--account=compute2-ris`**: `Invalid account or account/partition combination specified`.
- **`-A=...` after `--pty bash`**: `You must specify an account`.
- **Hostname `c2-login-00N`**: `00N` is a placeholder in the RIS docs; use `001`.
- **A directory at `~/.ssh/id_ed25519`** (instead of a key file) made SSH report
  "bad permissions" and fall back to password prompts. Remove it and rerun `ssh-keygen`.

## Caveat

`OLLAMA_HOST=0.0.0.0` exposes the Ollama API, which has no authentication, to
other users on the cluster network while the job runs. They could use your GPU
time, but cannot read your data. Stop it when you are done.

## Batch job: the full Phase 3 experiment

`scripts/run_experiments_qwen.sbatch` runs the whole experiment on the cluster,
so the laptop and tunnel aren't needed. It starts Ollama on the job's GPU,
listening on `127.0.0.1` only, and runs `scripts/run_experiments.py` on the same
node. The full experiment is 1,400 runs (900 ordinary, 200 recovery, 300
variability).

**Time:** the 20-question check (2 workers, H100) took about 66 s, 96 s and
248 s per question for `direct_llm`, `single_agent` and `multi_agent`. At that
rate the full experiment is about 28 hours. The script uses 4 workers, which
should bring that down to roughly 15–20 hours (not yet measured). The job asks
for 36 hours, which covers the 2-worker estimate; `general-gpu` allows up to 15
days. A job that hits its limit loses only the questions in progress, because
results are saved every 10 questions and the next job resumes.

### One-time setup (login node)

```bash
git clone -b fix/boolean-closing-verdict https://github.com/Akash-Jairam/CSE-5106-FHIR-Multi-Agent-System.git
cd CSE-5106-FHIR-Multi-Agent-System
python3 --version                       # needs 3.12; try `module avail python` if not
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p logs
```

From the laptop, copy the settings and credentials it needs:

```bash
cd ~/Downloads/Git/CSE-5106-FHIR-Multi-Agent-System
host=akashj@c2-login-001.ris.wustl.edu
repo=CSE-5106-FHIR-Multi-Agent-System

# A config.yml with only the FHIR store settings; Qwen needs no API keys
{ echo 'MODEL: "qwen3:32b"'; sed -n '/^FHIR_CONFIG:/,$p' config.yml; } > /tmp/config.cluster.yml
scp -i ~/.ssh/id_ed25519 /tmp/config.cluster.yml $host:$repo/config.yml && rm /tmp/config.cluster.yml

# Google credentials for the FHIR store
ssh -i ~/.ssh/id_ed25519 $host 'mkdir -p ~/.config/gcloud && chmod 700 ~/.config/gcloud'
scp -i ~/.ssh/id_ed25519 ~/.config/gcloud/application_default_credentials.json $host:.config/gcloud/
ssh -i ~/.ssh/id_ed25519 $host 'chmod 600 ~/.config/gcloud/application_default_credentials.json'

# Optional: the tool cache (about 400 MB) saves repeat FHIR calls
rsync -a -e "ssh -i ~/.ssh/id_ed25519" .tool_cache/ $host:$repo/.tool_cache/
```

The credentials file holds a refresh token for your Google account, so anyone
who can read it can act as you on Google Cloud. A service account key with only
the Healthcare FHIR read role limits that, if the project allows one.

Before the first long job, check FHIR access from the login node:

```bash
.venv/bin/python scripts/check_setup.py --model qwen3:32b
```

The Qwen endpoint check will fail there, because Ollama only runs inside the job;
the FHIR checks should pass.

### Submit and monitor

```bash
sbatch scripts/run_experiments_qwen.sbatch                  # every group
sbatch scripts/run_experiments_qwen.sbatch --arms ordinary  # or just some
squeue -u $USER -o "%.10i %.12P %.8T %.10M %.10L %N"
tail -f logs/phase3-qwen-<jobid>.out
```

To run past the time limit, queue follow-up jobs that start only after the
previous one ends, then resubmit the same command until `status` shows every arm done:

```bash
sbatch --dependency=singleton scripts/run_experiments_qwen.sbatch
```

Results go to `output/experiments/qwen3:32b/held_out/`. Score them with
`python scripts/run_experiments.py analyze --model qwen3:32b --base_url http://127.0.0.1:11434/v1 --max_total_tokens 200000 --max_tool_calls 30 --split held_out`,
or copy the folder back to the laptop with `rsync` and run `analyze` there with
the same arguments.

If a 36-hour job waits for days, shorter jobs start sooner: Slurm can fit them
into gaps. Lower a pending job's limit, allow the preemptible partition, and
chain follow-ups (each resumes where the last stopped):

```bash
scontrol update JobId=<jobid> TimeLimit=8:00:00 Partition=general-gpu,general-preempt-gpu
for i in 1 2 3; do
  sbatch --time=8:00:00 --partition=general-gpu,general-preempt-gpu --dependency=singleton \
         scripts/run_experiments_qwen.sbatch
done
```

### Other models and splits

The script takes `MODEL`, `SPLIT` and `OLLAMA_MODELS` from the environment, and
pulls the model on the GPU node if it is missing. `gpt-oss:120b` (65 GB) does not
fit the 47 GB home quota, so keep it on the lab's scratch space, and give the job
more memory for loading it. Give each model its own job name, so `singleton`
chains don't mix models:

```bash
MODEL=gpt-oss:120b OLLAMA_MODELS=/scratch2/fs1/di2cap/$USER/ollama-models \
  sbatch --job-name=phase3-gpt-oss --mem=128G scripts/run_experiments_qwen.sbatch
SPLIT=dev sbatch --job-name=phase3-qwen-dev scripts/run_experiments_qwen.sbatch --arms ordinary_multi_agent
```

Results go to `output/experiments/<model>/<split>/`. To compare release rules for
multi_agent's verification gate on saved results, see `scripts/release_policy.py`.

### Serving gpt-oss with vLLM

Ollama 0.40.13 rejects a gpt-oss reply whose tool-call arguments are not valid
JSON with HTTP 500, so the call never reaches the agent; in the Ollama run this
failed 50 single_agent and 103 multi_agent questions on the first attempt.
`scripts/run_experiments_vllm.sbatch` serves gpt-oss with vLLM instead, which
returns such a call, and the agents send the model a tool error as they do for
any other model. It keeps the 32k context and uses port 8000, so its results
never mix with the Ollama run's.

One-time setup on the login node (vLLM and the weights go on scratch, about 75 GB):

```bash
S=/scratch2/fs1/di2cap/$USER
/cm/local/apps/python312/bin/python3 -m venv $S/vllm-venv
$S/vllm-venv/bin/pip install vllm==0.33.0
HF_HOME=$S/hf $S/vllm-venv/bin/hf download openai/gpt-oss-120b --exclude "original/*" "metal/*"
```

Run it from a separate checkout, so a model already running from the main copy
keeps its code fingerprint:

```bash
git worktree add ~/fhir-gptoss-vllm <commit>
cd ~/fhir-gptoss-vllm && ln -s ~/CSE-5106-FHIR-Multi-Agent-System/.venv .venv
cp ~/CSE-5106-FHIR-Multi-Agent-System/config.yml . && mkdir -p logs
for i in 1 2 3; do
  sbatch --job-name=phase3-gpt-oss-vllm --dependency=singleton scripts/run_experiments_vllm.sbatch
done
SPLIT=dev sbatch --job-name=phase3-gpt-oss-vllm --dependency=singleton --time=2:00:00 \
  scripts/run_experiments_vllm.sbatch --arms ordinary_multi_agent
```

vLLM's own log is `logs/vllm-<jobid>.log`.
