#!/bin/bash

cd $PROJECT_BASE_PATH/scripts/eval

MODEL_PATH=$PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/no_warmup/sml/SmolLM2-135M-20B-sml-bs4096
sh lm_eval_summary.sh $MODEL_PATH

MODEL_PATH=$PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/no_warmup/sml_mask/SmolLM2-135M-20B-sml_mask-bs4096
sh lm_eval_summary.sh $MODEL_PATH

MODEL_PATH=$PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/no_warmup/sml_ent_id/SmolLM2-135M-20B-core_ent_id-bs4096
sh lm_eval_summary.sh $MODEL_PATH

MODEL_PATH=$PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/no_warmup/sml_rnd_id/SmolLM2-135M-20B-rnd_id-bs4096
sh lm_eval_summary.sh $MODEL_PATH
