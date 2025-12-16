# uv run python generate_sft_data_from_json.py /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/sft /home/pj25000107/ku50001566/projects/knowledge_decoupling/input/tokenized/sft/qwen/mix Qwen/Qwen3-0.6B-Base
echo ">>> mix smollm2"
uv run python generate_sft_data_from_json.py /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/sft /home/pj25000107/ku50001566/projects/knowledge_decoupling/input/tokenized/sft/smollm2/mix HuggingFaceTB/SmolLM2-135M
echo ">>> squad_v2_ctxt qwen"
uv run python generate_sft_data_from_json.py /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/sft/squad_v2_ctxt.json /home/pj25000107/ku50001566/projects/knowledge_decoupling/input/tokenized/sft/qwen/squad_v2_ctxt Qwen/Qwen3-0.6B-Base
echo ">>> squad_v2_ctxt smollm2"
uv run python generate_sft_data_from_json.py /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/sft/squad_v2_ctxt.json /home/pj25000107/ku50001566/projects/knowledge_decoupling/input/tokenized/sft/smollm2/squad_v2_ctxt HuggingFaceTB/SmolLM2-135M