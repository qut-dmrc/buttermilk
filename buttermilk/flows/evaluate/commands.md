base_run_name=moderate_drag_20240808_1802

# Test and evaluate variant_0:
# Test-runs
pf run create --flow ./moderate --data /tmp/osb_drag_toxic_train.jsonl --column-mapping source='${data.source}' text='${data.text}' alt_text='${data.alt_text}' datasets='["drag queens"]' use_alt_text=false run_local_models=false --variant '${moderate.perspective}' --name "${base_run_name}_0"
pf run create --flow ./moderate --data /tmp/osb_drag_toxic_train.jsonl --column-mapping source='${data.source}' text='${data.text}' alt_text='${data.alt_text}' datasets='["drag queens"]' use_alt_text=false run_local_models=false --variant '${moderate.comprehend}' --name "${base_run_name}_1"

pf run create --flow evaluate --data /tmp/test_vaw.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}' --run 3f832092-963e-4cbd-a85d-7161c7e5f395_prompt_0

eval_automod-flow-20240815T0356Z-7MBP-6c4962fcd0f9_vaw_claude3opus
automod-flow-20240815T0356Z-7MBP-6c4962fcd0f9_vaw_claude3opus
eval_automod-flow-20240815T0356Z-7MBP-6c4962fcd0f9_vaw_gpt4o
automod-flow-20240815T0356Z-7MBP-6c4962fcd0f9_vaw_gpt4o
automod-flow-20240815T0259Z-fxSG-6c4962fcd0f9_vaw_claude3opus
eval_automod-flow-20240815T0259Z-fxSG-6c4962fcd0f9_vaw_gpt4o
automod-flow-20240815T0259Z-fxSG-6c4962fcd0f9_vaw_gpt4o

# Evaluate-runs
pf run create --flow ./moderate_eval --data /tmp/osb_drag_toxic_train.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}'  --run "${base_run_name}_0" --name "eval_${base_run_name}_0"
pf run create --flow ./moderate_eval --data /tmp/osb_drag_toxic_train.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}' --run "${base_run_name}_1" --name "eval_${base_run_name}_1"

pf run create --flow automod/evaluate --data /tmp/osb_drag_toxic_train.jsonl --run judge_comprehend_20240809_114356_184184 --column-mapping groundtruth="${data.expected}" prediction="${run.outputs.result}" --name eval_comprehend_20240809_114356_18418

pfazure run list -r 25 | jq -cr ".[].name" | grep -v local|xargs -I % echo pf run create --flow ./moderate_eval --data /tmp/osb_drag_toxic_train.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}' --run '%' --name 'eval_%'

pf run create --flow ./moderate_eval --data /tmp/osb_drag_toxic_train.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}' --run moderate_llamaguard3hf_20240808_114954_576000 --name eval_moderate_llamaguard3hf_20240808_114954_576000n
pf run create --flow ./moderate_eval --data /tmp/osb_drag_toxic_train.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}' --run moderate_llamaguard3hfint8_20240808_114954_576000 --name eval_moderate_llamaguard3hfint8_20240808_114954_576000n
pf run create --flow ./moderate_eval --data /tmp/osb_drag_toxic_train.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}' --run moderate_llamaguard3octo_20240808_114954_576000 --name eval_moderate_llamaguard3octo_20240808_114954_576000n
pf run create --flow ./moderate_eval --data /tmp/osb_drag_toxic_train.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}' --run moderate_llamaguard2together_20240808_114954_576000 --name eval_moderate_llamaguard2together_20240808_114954_576000n
pf run create --flow ./moderate_eval --data /tmp/osb_drag_toxic_train.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}' --run moderate_llamaguard1together_20240808_114954_576000 --name eval_moderate_llamaguard1together_20240808_114954_576000n
pf run create --flow ./moderate_eval --data /tmp/osb_drag_toxic_train.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}' --run moderate_llamaguard1replicate_20240808_114954_576000 --name eval_moderate_llamaguard1replicate_20240808_114954_576000n
pf run create --flow ./moderate_eval --data /tmp/osb_drag_toxic_train.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}' --run moderate_llamaguard3together_20240808_114954_576000 --name eval_moderate_llamaguard3together_20240808_114954_576000n
pf run create --flow ./moderate_eval --data /tmp/osb_drag_toxic_train.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}' --run moderate_llamaguard2replicate_20240808_114954_576000 --name eval_moderate_llamaguard2replicate_20240808_114954_576000n
pf run create --flow ./moderate_eval --data /tmp/osb_drag_toxic_train.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}' --run moderate_azurecontentsafety_20240808_114954_576000 --name eval_moderate_azurecontentsafety_20240808_114954_576000n
pf run create --flow ./moderate_eval --data /tmp/osb_drag_toxic_train.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}' --run moderate_azuremoderator_20240808_114954_576000 --name eval_moderate_azuremoderator_20240808_114954_576000n
pf run create --flow ./moderate_eval --data /tmp/osb_drag_toxic_train.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}' --run moderate_openaimoderator_20240808_114954_576000 --name eval_moderate_openaimoderator_20240808_114954_576000n
pf run create --flow ./moderate_eval --data /tmp/osb_drag_toxic_train.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}' --run moderate_regard_20240808_114954_576000 --name eval_moderate_regard_20240808_114954_576000n
pf run create --flow ./moderate_eval --data /tmp/osb_drag_toxic_train.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}' --run moderate_lftw_20240808_114954_576000 --name eval_moderate_lftw_20240808_114954_576000n
pf run create --flow ./moderate_eval --data /tmp/osb_drag_toxic_train.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}' --run moderate_honest_20240808_114954_576000 --name eval_moderate_honest_20240808_114954_576000n
pf run create --flow ./moderate_eval --data /tmp/osb_drag_toxic_train.jsonl --column-mapping groundtruth='${data.expected}' result='${run.outputs.result}' --run moderate_perspective_20240808_114954_576000 --name eval_moderate_perspective_20240808_114954_576000m