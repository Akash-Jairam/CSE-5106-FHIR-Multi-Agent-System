# Demo interface (P3-23). Pass the experiment's settings after "--", e.g.
#   bash scripts/run_demo.sh --model gpt-5-nano --max_total_tokens 200000 --max_tool_calls 30
streamlit run demo/app.py --server.address localhost --client.toolbarMode minimal \
  --browser.gatherUsageStats false -- "$@"
