# qsub sft_llama_nonce_train_job.sh
# qsub sft_llama_hf_train_job.sh
# qsub sft_llama_sml_train_job.sh
qsub sft_smollm2_135M_hf_train_job.sh
qsub sft_smollm2_135M_nonce_train_job.sh
qsub sft_smollm2_135M_sml_train_job.sh
qsub sft_gpt2_hf_train_job.sh
qsub sft_gpt2_nonce_train_job.sh
qsub sft_gpt2_sml_train_job.sh
watch -d qstat