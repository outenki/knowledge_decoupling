echo
echo ">>> meta-llama/Llama-3.2-1B/SmolLM2-135M-20B-bs1024"
sh ./extract_lm_eval_results.sh meta-llama/Llama-3.2-1B/SmolLM2-135M-20B-bs1024|grep -v ">>>"|grep -v "N/A"|grep -v "blimp_"|grep -v "ewok_"|cut -d":" -f2
echo
echo ">>> meta-llama/Llama-3.2-1B/SmolLM2-135M-20B-core-bs1024"
sh ./extract_lm_eval_results.sh meta-llama/Llama-3.2-1B/SmolLM2-135M-20B-core-bs1024|grep -v ">>>"|grep -v "N/A"|grep -v "blimp_"|grep -v "ewok_"|cut -d":" -f2
echo
echo ">>> Qwen/Qwen2.5-0.5B/SmolLM2-135M-20B-bs1024"
sh ./extract_lm_eval_results.sh Qwen/Qwen2.5-0.5B/SmolLM2-135M-20B-bs1024|grep -v ">>>"|grep -v "N/A"|grep -v "blimp_"|grep -v "ewok_"|cut -d":" -f2
echo
echo ">>> Qwen/Qwen2.5-0.5B/SmolLM2-135M-20B-core-bs1024"
sh ./extract_lm_eval_results.sh Qwen/Qwen2.5-0.5B/SmolLM2-135M-20B-core-bs1024|grep -v ">>>"|grep -v "N/A"|grep -v "blimp_"|grep -v "ewok_"|cut -d":" -f2
echo
echo ">>> allenai/OLMo-2-0425-1B/SmolLM2-135M-20B-bs1024"
sh ./extract_lm_eval_results.sh allenai/OLMo-2-0425-1B/SmolLM2-135M-20B-bs1024|grep -v ">>>"|grep -v "N/A"|grep -v "blimp_"|grep -v "ewok_"|cut -d":" -f2
echo
echo ">>> allenai/OLMo-2-0425-1B/SmolLM2-135M-20B-core-bs1024"
sh ./extract_lm_eval_results.sh allenai/OLMo-2-0425-1B/SmolLM2-135M-20B-core-bs1024|grep -v ">>>"|grep -v "N/A"|grep -v "blimp_"|grep -v "ewok_"|cut -d":" -f2