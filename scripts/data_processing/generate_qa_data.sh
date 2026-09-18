#!/bin/bash


# echo ">>> QASC"
# uv run python generate_qa_data.py -dn qasc -p -o $OUTPUT_PATH/qasc
# echo ">>> mintaka"
# uv run python generate_qa_data.py -dn mintaka -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/mintaka/data -o $OUTPUT_PATH/mintaka
# echo ">>> mintaka_multihop"
# uv run python generate_qa_data.py -dn mintaka -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/mintaka/data -o $OUTPUT_PATH/mintaka_multihop
# echo ">>> complex_web_questions"
# uv run python generate_qa_data.py -dn cwq -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/complexwebquestions_V1_1 -o $OUTPUT_PATH/cwq
# echo ">>> metaqa_1hop"
# uv run python generate_qa_data.py -dn metaqa -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/metaqa_ntm/metaqa_ntm_1hop -o $OUTPUT_PATH/metaqa_1hop
# echo ">>> metaqa_2hop"
# uv run python generate_qa_data.py -dn metaqa -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/metaqa_ntm/metaqa_ntm_2hop -o $OUTPUT_PATH/metaqa_2hop
# echo ">>> metaqa_3hop"
# uv run python generate_qa_data.py -dn metaqa -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/metaqa_ntm/metaqa_ntm_3hop -o $OUTPUT_PATH/metaqa_3hop
# echo ">>> google_re"
# uv run python generate_qa_data.py -dn google_re -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/Google_RE -o $OUTPUT_PATH/google_re_long_context -ck snippet
# uv run python generate_qa_data.py -dn google_re -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/Google_RE -o $OUTPUT_PATH/google_re_short_context -ck considered_sentences
# echo ">>> google_re_conflict"
# conflict as evaluate data
# uv run python generate_qa_data.py -cc mod -dn google_re -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/Google_RE_conflict -o $OUTPUT_PATH/google_re_long_context -ck snippet
# uv run python generate_qa_data.py -cc mod -dn google_re -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/Google_RE_conflict -o $OUTPUT_PATH/google_re_short_context -ck considered_sentences
# ori as ext training data 
# uv run python generate_qa_data.py -cc ori -dn google_re -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/Google_RE_conflict -o $EXT_TRAINING_PATH/google_re_long_context -ck snippet
# uv run python generate_qa_data.py -cc ori -dn google_re -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/Google_RE_conflict -o $EXT_TRAINING_PATH/google_re_short_context -ck considered_sentences
# echo ">>> google_re no context"
# uv run python generate_qa_data.py -dn google_re -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/Google_RE -o $OUTPUT_PATH/google_re_no_context -ck ""
# echo ">>> race"
# uv run python generate_qa_data.py -dn race -lp $PROJECT_BASE_PATH/data/race -o $OUTPUT_PATH/race -ck ""
# echo ">>> SquAD_based"
# uv run python generate_qa_data.py -dn based_squad -lp $PROJECT_BASE_PATH/data/based_squad -o $OUTPUT_PATH/based_squad -ck ""
# echo ">>> SquAD_v2"
# uv run python generate_qa_data.py -dn squadv2 --core-count -o $OUTPUT_PATH/jsonl/squadv2 -ot jsonl --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv -at 10 --split validation --ent-generator "ENT_ID" --unk-generator "UNK_ID"

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

echo
echo ">>> xsum"
uv run python generate_qa_data.py -dn xsum -o $OUTPUT_PATH/xsum -ot jsonl

echo
echo ">>> samsum"
uv run python generate_qa_data.py -dn samsum -o $OUTPUT_PATH/samsum -ot jsonl

echo
echo ">>> gigaword"
uv run python generate_qa_data.py -dn gigaword -o $OUTPUT_PATH/gigaword -ot jsonl

echo
echo ">>> mrpc"
uv run python generate_qa_data.py -dn mrpc -o $OUTPUT_PATH/mrpc -ot jsonl

echo
echo ">>> paws_en"
uv run python generate_qa_data.py -dn paws_en -o $OUTPUT_PATH/paws_en -ot jsonl

# echo ">>> triviaqa_rc_context"
# uv run python generate_qa_data.py -dn triviaqa_rc_context -o $OUTPUT_PATH/jsonl/triviaqa_rc_context -ot jsonl

# echo ">>> triviaqa_rc_nocontext"
# uv run python generate_qa_data.py -dn triviaqa_rc_nocontext -o $OUTPUT_PATH/jsonl/triviaqa_rc_nocontext -ot jsonl

# echo ">>> google_boolq"
# uv run python generate_qa_data.py  \
#     -dn boolq \
#     -o $OUTPUT_PATH/jsonl/google_boolq \
#     -ot jsonl

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

# echo ">>> google_boolq"
# uv run python generate_qa_data.py  \
#     -dn boolq \
#     -o $OUTPUT_PATH/jsonl/google_boolq \
#     -ot jsonl

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

# echo ">>> google_boolq"
# uv run python generate_qa_data.py  \
#     -dn boolq \
#     -o $OUTPUT_PATH/jsonl/google_boolq \
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