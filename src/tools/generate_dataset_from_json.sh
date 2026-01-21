BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
JSON_DIR=$BASE_PATH/data/ext/test

uv run python generate_dataset_from_json.py \
    -jf $JSON_DIR/qa_arc_easy.json \
    -jf $JSON_DIR/qa_arc_challenge.json \
    -jf $JSON_DIR/qasc.json \
    -jf $JSON_DIR/qa_boolq_ctxt.json \
    -jf $JSON_DIR/squad_v2_ctxt.json \
    -o $JSON_DIR
