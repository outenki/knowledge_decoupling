# # google_boolq
# uv run python extract_samples_with_replaced_tokens_num.py \
#     -s1 /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml/SmolLM2-135M-20B-sml-bs4096-sft_google_boolq_train/eval/google_boolq/samples_google_boolq_2026-09-23T00-50-24.042167.jsonl \
#     -s2 /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml_mask/SmolLM2-135M-20B-sml_mask-bs4096-sft_google_boolq_train/eval/google_boolq/samples_google_boolq_2026-09-23T01-20-37.045916.jsonl \
#     -rp /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/jsonl/jsonl/google_boolq_rnd_id_count/validation.jsonl \
#     -m acc\
#     -o /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/jsonl/jsonl/google_boolq_rnd_id_count/sml_mask.csv

# # squadv2
# uv run python extract_samples_with_replaced_tokens_num.py \
#     -s1 /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml/SmolLM2-135M-20B-sml-bs4096-sft_squadv2_train/eval/squadv2/samples_squadv2_2026-09-23T01-07-39.280724.jsonl \
#     -s2 /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml_mask/SmolLM2-135M-20B-sml_mask-bs4096-sft_squadv2_train/eval/squadv2/samples_squadv2_2026-09-23T01-31-45.426187.jsonl \
#     -rp /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/jsonl/jsonl/squadv2_rnd_id_count/validation.jsonl \
#     -m f1\
#     -o /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/jsonl/jsonl/squadv2_rnd_id_count/sml_mask.csv \

# # triviaqa_rc_context
# uv run python extract_samples_with_replaced_tokens_num.py \
#     -s1 /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml/SmolLM2-135M-20B-sml-bs4096-sft_triviaqa_rc_context_train/eval/triviaqa_rc_context/samples_triviaqa_rc_context_2026-09-23T19-08-00.374223.jsonl \
#     -s2 /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml_mask/SmolLM2-135M-20B-sml_mask-bs4096-sft_triviaqa_rc_context_train/eval/triviaqa_rc_context/samples_triviaqa_rc_context_2026-09-23T19-51-02.362847.jsonl \
#     -rp /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/jsonl/jsonl/triviaqa_rc_context_rnd_id_count/validation.jsonl \
#     -m em \
#     -o /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/jsonl/jsonl/triviaqa_rc_context_rnd_id_count/sml_mask.csv

# # arc_easy
# uv run python extract_samples_with_replaced_tokens_num.py \
#     -s1 /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml/SmolLM2-135M-20B-sml-bs4096-sft_arc_easy_train/eval/arc_easy/samples_arc_easy_2026-09-23T18-49-01.922965.jsonl \
#     -s2 /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml_mask/SmolLM2-135M-20B-sml_mask-bs4096-sft_arc_easy_train/eval/arc_easy/samples_arc_easy_2026-09-23T19-38-37.865384.jsonl \
#     -rp /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/jsonl/jsonl/arc_easy_count/test.jsonl \
#     -m acc\
#     -o /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/jsonl/jsonl/arc_easy_count/sml_mask.csv

# # arc_challenge
# uv run python extract_samples_with_replaced_tokens_num.py \
#     -s1 /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml/SmolLM2-135M-20B-sml-bs4096-sft_arc_challenge_train/eval/arc_challenge/samples_arc_challenge_2026-09-23T18-50-36.582697.jsonl \
#     -s2 /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml_mask/SmolLM2-135M-20B-sml_mask-bs4096-sft_arc_challenge_train/eval/arc_challenge/samples_arc_challenge_2026-09-23T19-40-10.625262.jsonl \
#     -rp /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/jsonl/jsonl/arc_challenge_count/test.jsonl \
#     -m acc\
#     -o /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/jsonl/jsonl/arc_challenge_count/sml_mask.csv

# triviaqa_rc_nocontext
uv run python extract_samples_with_replaced_tokens_num.py \
    -s1 /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml/SmolLM2-135M-20B-sml-bs4096-sft_triviaqa_rc_nocontext_train/eval/triviaqa_rc_nocontext/samples_triviaqa_2026-09-23T01-01-28.894863.jsonl \
    -s2 /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml_mask/SmolLM2-135M-20B-sml_mask-bs4096-sft_triviaqa_rc_nocontext_train/eval/triviaqa_rc_nocontext/samples_triviaqa_2026-09-23T01-31-37.451801.jsonl \
    -rp /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/jsonl/jsonl/triviaqa_rc_nocontext_count/validation.jsonl \
    -m em \
    -o /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/jsonl/jsonl/triviaqa_rc_nocontext_count/sml_mask.csv

# commonsense_qa
# uv run python extract_samples_with_replaced_tokens_num.py \
#     -s1 /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml/SmolLM2-135M-20B-sml-bs4096-sft_commonsense_qa_train/eval/commonsense_qa/samples_commonsense_qa_2026-09-23T18-51-59.933013.jsonl \
#     -s2 /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml_mask/SmolLM2-135M-20B-sml_mask-bs4096-sft_commonsense_qa_train/eval/commonsense_qa/samples_commonsense_qa_2026-09-23T19-41-40.262120.jsonl \
#     -rp /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/jsonl/jsonl/commonsense_qa_count/validation.jsonl \
#     -m acc\
#     -o /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/jsonl/jsonl/commonsense_qa_count/sml_mask.csv


# winogrande
# uv run python extract_samples_with_replaced_tokens_num.py \
#     -s1 /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml/SmolLM2-135M-20B-sml-bs4096-sft_winogrande_train/eval/winogrande/samples_winogrande_2026-09-23T18-53-31.083790.jsonl \
#     -s2 /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml_mask/SmolLM2-135M-20B-sml_mask-bs4096-sft_winogrande_train/eval/winogrande/samples_winogrande_2026-09-23T19-43-07.133983.jsonl\
#     -rp /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/jsonl/jsonl/winogrande_count/validation.jsonl \
#     -m acc\
#     -o /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/jsonl/jsonl/winogrande_count/sml_mask.csv

# piqa
# uv run python extract_samples_with_replaced_tokens_num.py \
#     -s1 /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml/SmolLM2-135M-20B-sml-bs4096-sft_piqa_train/eval/piqa/samples_piqa_2026-09-23T18-55-08.433024.jsonl \
#     -s2 /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/meta-llama/Llama-3.2-1B/no_warmup/sml_mask/SmolLM2-135M-20B-sml_mask-bs4096-sft_piqa_train/eval/piqa/samples_piqa_2026-09-23T19-44-36.913405.jsonl\
#     -rp /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/jsonl/jsonl/piqa_count/validation.jsonl \
#     -m acc\
#     -o /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/jsonl/jsonl/piqa_count/sml_mask.csv