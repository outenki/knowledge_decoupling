# MODEL_PATH=/lustre1/work/c30897/wtq/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml/SmolLM2-135M-20B-sml-bs4096
# echo "EValuating google_re for: $MODEL_PATH"
# sh lm_eval_google_re.sh "$MODEL_PATH"

# MODEL_PATH=/lustre1/work/c30897/wtq/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml/SmolLM2-135M-20B-sml-bs4096-ext_google_re_mix_short_test-bs4096
# echo "EValuating google_re for: $MODEL_PATH"
# sh lm_eval_google_re.sh "$MODEL_PATH"

# MODEL_PATH=/lustre1/work/c30897/wtq/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml_mask/SmolLM2-135M-20B-sml_mask-bs4096
# echo "EValuating google_re for: $MODEL_PATH"
# sh lm_eval_google_re.sh "$MODEL_PATH"

# MODEL_PATH=/lustre1/work/c30897/wtq/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml_mask/SmolLM2-135M-20B-sml_mask-bs4096-ext_google_re_mix_short_test-mask-bs4096
# echo "EValuating google_re for: $MODEL_PATH"
# sh lm_eval_google_re.sh "$MODEL_PATH"

MODEL_PATH=/lustre1/work/c30897/wtq/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/rnd/rnd
echo "EValuating google_re for: $MODEL_PATH"
sh lm_eval_google_re.sh "$MODEL_PATH"

MODEL_PATH=/lustre1/work/c30897/wtq/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/rnd/rnd-ext_google_re_mix_short_test-bs4096
echo "EValuating google_re for: $MODEL_PATH"
sh lm_eval_google_re.sh "$MODEL_PATH"