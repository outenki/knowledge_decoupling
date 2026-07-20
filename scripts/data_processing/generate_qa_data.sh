#!/bin/bash
OUTPUT_PATH=$PROJECT_BASE_PATH/input/evaluate_data/json
EXT_TRAINING_PATH=$PROJECT_BASE_PATH/data/ext
SFT_TRAINING_PATH=$PROJECT_BASE_PATH/data/sft
# echo ">>> ARC-Easy"
# uv run python generate_qa_data.py -dn ai2_arc -sn ARC-Easy -p -o $OUTPUT_PATH/arc_easy
# echo ">>> ARC-Challenge"
# uv run python generate_qa_data.py -dn ai2_arc -sn ARC-Challenge -p -o $OUTPUT_PATH/arc_challenge
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
# echo ">>> commonsense_qa"
# uv run python generate_qa_data.py -dn commonsense_qa  -o $OUTPUT_PATH/commonsense_qa
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
# uv run python generate_qa_data.py -dn squadv2 -o $OUTPUT_PATH/squadv2

# echo ">>> BOOLQ"
# uv run python generate_qa_data.py -dn boolq -o $OUTPUT_PATH/boolq

# echo ">>> triviaqa_rc_context"
# uv run python generate_qa_data.py -dn triviaqa_rc_context -o $OUTPUT_PATH/triviaqa_rc_context

echo ">>> google_boolq_core"
uv run python generate_qa_data.py -dn google_boolq_core -lp /home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/json/google_boolq_core -o $OUTPUT_PATH/google_boolq_core --aoa $PROJECT_BASE_PATH/data/core/AOA/aoa.csv -at 10
# echo ">>> squadv2_core"
# uv run python generate_qa_data.py -dn squadv2 -rc -o $OUTPUT_PATH/squadv2_core --aoa $PROJECT_BASE_PATH/data/core/AOA/aoa.csv -at 10
# echo ">>> triviaqa_rc_core"
# uv run python generate_qa_data.py -dn triviaqa_rc_context -rc -o $OUTPUT_PATH/triviaqa_rc_context_core --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv -at 10