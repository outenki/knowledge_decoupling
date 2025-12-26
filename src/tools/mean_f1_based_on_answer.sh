#!/bin/bash

BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling/output/HuggingFaceTB/SmolLM2-135M
EVAL_DATA="squad_v2_ctxt"

#  gpt2
for m in random hf hf-sft/mix-e3 hf-sft/squad_v2_ctxt-e3; do
    echo ">>>>>> $m"
    uv run python mean_f1_based_on_answer.py \
        $BASE_PATH/$m/evaluation_20251215/generation/0_shots/$EVAL_DATA/evaluated_samples.json "i don't know."
done