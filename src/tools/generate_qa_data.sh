#!/bin/bash
OUTPUT_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling/input/evaluate_data/unformated/
# echo ">>> ARC-Easy"
# uv run python generate_qa_data.py -dn ai2_arc -sn ARC-Easy -p -o $OUTPUT_PATH/arc_easy
# echo ">>> ARC-Challenge"
# uv run python generate_qa_data.py -dn ai2_arc -sn ARC-Challenge -p -o $OUTPUT_PATH/arc_challenge
# echo ">>> QASC"
# uv run python generate_qa_data.py -dn qasc -p -o $OUTPUT_PATH/qasc
# echo ">>> BOOLQ"
# uv run python generate_qa_data.py -dn boolq -p -o $OUTPUT_PATH/boolq
# echo ">>> SquAD_v2"
# uv run python generate_qa_data.py -dn squad_v2 -p -o $OUTPUT_PATH/squad_v2
# echo ">>> mintaka"
# uv run python generate_qa_data.py -dn mintaka -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/mintaka/data -o $OUTPUT_PATH/mintaka
echo ">>> mintaka_multihop"
uv run python generate_qa_data.py -dn mintaka -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/mintaka/data -o $OUTPUT_PATH/mintaka_multihop
# echo ">>> complex_web_questions"
# uv run python generate_qa_data.py -dn cwq -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/complexwebquestions_V1_1 -o $OUTPUT_PATH/cwq
# echo ">>> metaqa_1hop"
# uv run python generate_qa_data.py -dn metaqa -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/metaqa_ntm/metaqa_ntm_1hop -o $OUTPUT_PATH/metaqa_1hop
# echo ">>> metaqa_2hop"
# uv run python generate_qa_data.py -dn metaqa -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/metaqa_ntm/metaqa_ntm_2hop -o $OUTPUT_PATH/metaqa_2hop
# echo ">>> metaqa_3hop"
# uv run python generate_qa_data.py -dn metaqa -lp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/metaqa_ntm/metaqa_ntm_3hop -o $OUTPUT_PATH/metaqa_3hop