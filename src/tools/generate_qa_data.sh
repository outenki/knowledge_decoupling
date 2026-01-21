#!/bin/bash
OUTPUT_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling/input/evaluate_data/unformated_probing/
echo ">>> ARC-Easy"
uv run python generate_qa_data.py -dn ai2_arc -sn ARC-Easy -p -o $OUTPUT_PATH/arc_easy
echo ">>> ARC-Challenge"
uv run python generate_qa_data.py -dn ai2_arc -sn ARC-Challenge -p -o $OUTPUT_PATH/arc_challenge
echo ">>> QASC"
uv run python generate_qa_data.py -dn qasc -p -o $OUTPUT_PATH/qasc
echo ">>> BOOLQ"
uv run python generate_qa_data.py -dn boolq -p -o $OUTPUT_PATH/boolq
echo ">>> SquAD_v2"
uv run python generate_qa_data.py -dn squad_v2 -p -o $OUTPUT_PATH/squad_v2