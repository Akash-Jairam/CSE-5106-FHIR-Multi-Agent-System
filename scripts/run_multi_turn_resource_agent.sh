model="$(python utils/model_config.py)"  # $MODEL, then MODEL in config.yml, then gpt-6-luna
agent_strategy="multi_turn_resource"
python run_agent.py \
  --agent_strategy $agent_strategy \
  --model $model \
  --input final_dataset/questions_answers_sql_fhir.csv \
  --output output/${agent_strategy}_${model}_results.json \
  --num_processes 1