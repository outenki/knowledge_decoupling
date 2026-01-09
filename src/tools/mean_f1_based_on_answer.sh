#!/bin/bash

BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling/output/gpt2
EVAL_DATA="squad_v2"

#  gpt2
for m in \
    random-sft-mix_qa_without_options-ep3 \
    hf-sft-mix_qa_without_options-ep3 \
    nonce/smolLM2-nonce-mn3-bs1024-dl0-ep1-sft-mix_qa_without_options-ep3 \
    smolLM2/smolLM2-bs1024-dl0-ep1-sft-mix_qa_without_options-ep3; do
    echo ">>>>>> $m"
    uv run python mean_f1_based_on_answer.py \
        $BASE_PATH/$m/evaluation_20260105/generation/0_shots/$EVAL_DATA/evaluated_samples.json "i don't know."
done