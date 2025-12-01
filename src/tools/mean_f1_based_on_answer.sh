EVAL_DATA="squad_v2"
#  gpt2
for m in gpt2-hf gpt2-sft_squad_v2_ctxt-ep3 gpt2-sft_mix-ep3; do
    echo ">>>>>> $m"
    uv run python mean_f1_based_on_answer.py /home/pj25000107/ku50001566/projects/knowledge_decoupling/output/gpt2/$m/evaluation_20251127_simple/generation/0_shots/$EVAL_DATA/evaluated_samples.json "i don't know."
done

echo
#  nonce
    # smolLM2-nonce-bs1024-dl0-ep1 \
    # smolLM2-nonce-bs1024-dl0-ep1-sft_mix-e3 \
    # smolLM2-nonce-mn3-bs1024-dl0-ep1 \
    # smolLM2-nonce-mn3-bs1024-dl0-ep1-sft_mix-e3 \
    # smolLM2-ox3000-bs1024-dl0-ep3 \
    # smolLM2-ox3000-bs1024-dl0-ep3-sft_squad_v2_ctxt-ep3 \
    # smolLM2-ox3000-bs1024-dl0-ep3-sft_mix-ep3 \
    # smolLM2-bs1024-dl0-ep1 \
    # smolLM2-bs1024-dl0-ep1-sft_squad_v2_ctxt-ep3 \
    # smolLM2-bs1024-dl0-ep1-sft_mix-ep3
    # smolLM2-nonce-mn3-bs1024-dl0-ep1-sft_squad_v2_ctxt-e3
for m in \
    smolLM2-nonce-bs1024-dl0-ep1-sft_squad_v2_ctxt-e3
do
    echo ">>>>>> $m"
    uv run python mean_f1_based_on_answer.py /home/pj25000107/ku50001566/projects/knowledge_decoupling/output/gpt2/smolLM2/$m/evaluation_20251127_simple/generation/0_shots/$EVAL_DATA/evaluated_samples.json "i don't know." 
done