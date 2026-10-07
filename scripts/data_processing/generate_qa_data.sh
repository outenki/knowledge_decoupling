#!/bin/bash


OUTPUT_PATH=$PROJECT_BASE_PATH/input/evaluate_data/jsonl
EXT_TRAINING_PATH=$PROJECT_BASE_PATH/data/ext
SFT_TRAINING_PATH=$PROJECT_BASE_PATH/data/sft



# echo
# echo ">>> ARC-Easy"
# uv run python generate_qa_data.py -dn arc_easy -o $OUTPUT_PATH/arc_easy -ot jsonl

# echo
# echo ">>> ARC-Challenge"
# uv run python generate_qa_data.py -dn arc_challenge -o $OUTPUT_PATH/arc_challenge -ot jsonl

# echo
# echo ">>> commonsense_qa"
# uv run python generate_qa_data.py -dn commonsense_qa  -o $OUTPUT_PATH/commonsense_qa -ot jsonl

# echo
# echo ">>> winogrande"
# uv run python generate_qa_data.py -dn winogrande -o $OUTPUT_PATH/winogrande -ot jsonl

# echo
# echo ">>> piqa"
# uv run python generate_qa_data.py -dn piqa -o $OUTPUT_PATH/piqa -ot jsonl

# echo
# echo ">>> cnn_dailymail"
# uv run python generate_qa_data.py -dn cnn_dailymail -o $OUTPUT_PATH/cnn_dailymail -ot jsonl

# echo
# echo ">>> xsum"
# uv run python generate_qa_data.py -dn xsum -o $OUTPUT_PATH/xsum -ot jsonl

# echo
# echo ">>> samsum"
# uv run python generate_qa_data.py -dn samsum -o $OUTPUT_PATH/samsum -ot jsonl

# echo
# echo ">>> gigaword"
# uv run python generate_qa_data.py -dn gigaword -o $OUTPUT_PATH/gigaword -ot jsonl

# echo
# echo ">>> mrpc"
# uv run python generate_qa_data.py -dn mrpc -o $OUTPUT_PATH/mrpc -ot jsonl

# echo
# echo ">>> paws_en"
# uv run python generate_qa_data.py -dn paws_en -o $OUTPUT_PATH/paws_en -ot jsonl

# echo ">>> triviaqa_rc_context"
# uv run python generate_qa_data.py -dn triviaqa_rc_context -o $OUTPUT_PATH/jsonl/triviaqa_rc_context -ot jsonl

# echo ">>> triviaqa_rc_nocontext"
# uv run python generate_qa_data.py -dn triviaqa_rc_nocontext -o $OUTPUT_PATH/jsonl/triviaqa_rc_nocontext -ot jsonl

# echo ">>> google_boolq"
# uv run python generate_qa_data.py  \
#     -dn boolq \
#     -o $OUTPUT_PATH/jsonl/google_boolq \
#     -ot jsonl

# echo ">>> squadv2_rnd_id"
# uv run python generate_qa_data.py \
#     -dn squadv2 \
#     -o $OUTPUT_PATH/jsonl/squadv2_rnd_id \
#     --core-replace \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --ent-generator "ID" \
# echo ">>> triviaqa_rc_context"
# uv run python generate_qa_data.py -dn triviaqa_rc_context -o $OUTPUT_PATH/jsonl/triviaqa_rc_context -ot jsonl

# echo ">>> triviaqa_rc_nocontext"
# uv run python generate_qa_data.py -dn triviaqa_rc_nocontext -o $OUTPUT_PATH/jsonl/triviaqa_rc_nocontext -ot jsonl

# echo ">>> squadv2_core"
# uv run python generate_qa_data.py \
#     -dn squadv2 \
#     -o $OUTPUT_PATH/jsonl/squadv2_rnd_id \
#     --core-replace \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --ent-generator "ID" \
# echo ">>> triviaqa_rc_context"
# uv run python generate_qa_data.py -dn triviaqa_rc_context -o $OUTPUT_PATH/jsonl/triviaqa_rc_context -ot jsonl

# echo ">>> triviaqa_rc_nocontext"
# uv run python generate_qa_data.py -dn triviaqa_rc_nocontext -o $OUTPUT_PATH/jsonl/triviaqa_rc_nocontext -ot jsonl

# echo ">>> google_boolq_rnd_id"
# uv run python generate_qa_data.py  \
#     -dn boolq \
#     -o $OUTPUT_PATH/jsonl/google_boolq_rnd_id \
#     --core-replace \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-count \
#     --core-delimiter "<>" \
#     -ot jsonl

# echo ">>> arc_easy_count"
# uv run python generate_qa_data.py \
#     -dn arc_easy \
#     -o $OUTPUT_PATH/jsonl/arc_easy_count \
#     --core-replace \
#     --core-count \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --split test \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-delimiter "<>" \
#     -ot jsonl

# echo ">>> arc_challenge_count"
# uv run python generate_qa_data.py \
#     -dn arc_challenge \
#     -o $OUTPUT_PATH/jsonl/arc_challenge_count \
#     --core-replace \
#     --core-count \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --split test \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-delimiter "<>" \
#     -ot jsonl

# echo ">>> rc_nocontext_count"
# uv run python generate_qa_data.py \
#     -dn triviaqa_rc_nocontext \
#     -o $OUTPUT_PATH/jsonl/triviaqa_rc_nocontext_count \
#     --core-replace \
#     --core-count \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --split test \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-delimiter "<>" \
#     -ot jsonl

# echo ">>> google_boolq_rnd_id_count"
# uv run python generate_qa_data.py  \
#     -dn boolq \
#     -o $OUTPUT_PATH/jsonl/google_boolq_rnd_id_count \
#     --core-replace \
#     --core-count \
#     --split validation \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-count \
#     --core-delimiter "<>" \
#     -ot jsonl

# echo ">>> squadv2_rnd_id_count"
# uv run python generate_qa_data.py \
#     -dn squadv2 \
#     -o $OUTPUT_PATH/jsonl/squadv2_rnd_id_count \
#     --core-replace \
#     --core-count \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --split validation \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-delimiter "<>" \
#     -ot jsonl

# echo ">>> squadv2_rnd_id_count"
# uv run python generate_qa_data.py \
#     -dn squadv2 \
#     -o $OUTPUT_PATH/jsonl/squadv2_rnd_id_count \
#     --core-replace \
#     --core-count \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --split test \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-delimiter "<>" \
#     -ot jsonl

# echo ">>> commonsenseqa_rnd_id_count"
# uv run python generate_qa_data.py \
#     -dn commonsense_qa \
#     -o $OUTPUT_PATH/jsonl/commonsense_qa_count \
#     --core-replace \
#     --core-count \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --split test \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-delimiter "<>" \
#     -ot jsonl
# uv run python generate_qa_data.py \
#     -dn commonsense_qa \
#     -o $OUTPUT_PATH/jsonl/commonsense_qa_count \
#     --core-replace \
#     --core-count \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --split validation \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-delimiter "<>" \
#     -ot jsonl

# echo ">>> ewok_rnd_id_count"
# uv run python generate_qa_data.py \
#     -dn ewok \
#     -o $OUTPUT_PATH/jsonl/ewok_count \
#     --core-replace \
#     --core-count \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --split test \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-delimiter "<>" \
#     -ot jsonl
# uv run python generate_qa_data.py \
#     -dn ewok \
#     -o $OUTPUT_PATH/jsonl/ewok_count \
#     --core-replace \
#     --core-count \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --split validation \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-delimiter "<>" \
#     -ot jsonl

# echo ">>> winogrande_rnd_id_count"
# uv run python generate_qa_data.py \
#     -dn winogrande \
#     -o $OUTPUT_PATH/jsonl/winogrande_count \
#     --core-replace \
#     --core-count \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --split test \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-delimiter "<>" \
#     -ot jsonl
# uv run python generate_qa_data.py \
#     -dn winogrande \
#     -o $OUTPUT_PATH/jsonl/winogrande_count \
#     --core-replace \
#     --core-count \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --split validation \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-delimiter "<>" \
#     -ot jsonl

# echo ">>> piqa_rnd_id_count"
# uv run python generate_qa_data.py \
#     -dn piqa \
#     -o $OUTPUT_PATH/jsonl/piqa_count \
#     --core-replace \
#     --core-count \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --split test \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-delimiter "<>" \
#     -ot jsonl
# uv run python generate_qa_data.py \
#     -dn piqa \
#     -o $OUTPUT_PATH/jsonl/piqa_count \
#     --core-replace \
#     --core-count \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --split validation \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-delimiter "<>" \
#     -ot jsonl


# echo ">>> triviaqa_rc_context_rnd_id_count"
# uv run python generate_qa_data.py \
#     -dn triviaqa_rc_context \
#     -o $OUTPUT_PATH/jsonl/triviaqa_rc_context_rnd_id_count \
#     --core-replace \
#     --core-count \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --split validation \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-delimiter "<>" \
#     -ot jsonl

# echo ">>> triviaqa_rc_context_rnd_id_count"
# uv run python generate_qa_data.py \
#     -dn triviaqa_rc_context \
#     -o $OUTPUT_PATH/jsonl/triviaqa_rc_context_rnd_id_count \
#     --core-replace \
#     --core-count \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --split test \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-delimiter "<>" \
#     -ot jsonl


# echo ">>> squadv2_core"
# uv run python generate_qa_data.py \
#     -dn squadv2 \
#     -o $OUTPUT_PATH/jsonl/squadv2_rnd_id \
#     --core-replace \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-count \
#     --core-delimiter "<>" \
#     -ot jsonl

# echo ">>> triviaqa_rc_context_core"
# uv run python generate_qa_data.py \
#     -dn triviaqa_rc_context \
#     -o $OUTPUT_PATH/jsonl/triviaqa_rc_rnd_id \
#     --core-replace \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-count \
#     --core-delimiter "<>" \
#     -ot jsonl

# echo ">>> triviaqa_rc_nocontext_core"
# uv run python generate_qa_data.py \
#     -dn triviaqa_rc_nocontext \
#     -o $OUTPUT_PATH/jsonl/triviaqa_rc_nocontext_rnd_id \
#     --core-replace \
#     --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --ent-generator "ID" \
#     --unk-generator "ID" \
#     --core-count \
#     --core-delimiter "<>" \
#     -ot jsonl

# echo ">>> winogrande"
# uv run python generate_qa_data.py -dn winogrande -o $OUTPUT_PATH/jsonl/winogrande -ot jsonl

echo ">>> google_re_mix_conflict_short"
uv run python generate_qa_data.py \
    -dn google_re_mix_conflict_short \
    -lp /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/json/unformated/bak/google_re_conflict_short_context \
    -o $OUTPUT_PATH/google_re_mix_conflict_short \
    -ot jsonl

# echo ">>> google_re_mix_short"
# uv run python generate_qa_data.py \
#     -dn google_re_mix_short \
#     -lp /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/json/unformated/bak/google_re_short_context \
#     -o $OUTPUT_PATH/google_re_mix_short \
#     -ot jsonl
