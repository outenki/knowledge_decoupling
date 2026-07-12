#!/bin/bash
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_PATH=$PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/SmolLM2-135M-20B-bs1024
echo
echo ">>> Evaluating QA for: $MODEL_PATH"
sh llm_eval_qa.sh $MODEL_PATH
echo
echo ">>> Evaluating CONTEXT QA for: $MODEL_PATH"
sh llm_eval_context_qa.sh $MODEL_PATH
echo
echo ">>> Evaluating BLIMP for: $MODEL_PATH"
sh llm_eval_blimp.sh $MODEL_PATH