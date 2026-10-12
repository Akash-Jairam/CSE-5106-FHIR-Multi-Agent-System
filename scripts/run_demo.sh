#!/usr/bin/env bash
# Demo interface (P3-23). Pass the experiment's settings, e.g.
#   bash scripts/run_demo.sh --model gpt-5-nano --max_total_tokens 200000 --max_tool_calls 30
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec streamlit run demo/app.py --server.address localhost --client.toolbarMode minimal \
  --browser.gatherUsageStats false -- "$@"
