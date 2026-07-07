#! /bin/bash 

# for i in {0..10}; do
#     echo "Submitting job for tokenize_smolLM2_job_${i}.sh"
#     pjsub tokenize_smolLM2_job_${i}.sh
# done

# for i in {0..10}; do
#     echo "Submitting job for tokenize_smolLM2_nonce_job_${i}.sh"
#     pjsub tokenize_smolLM2_nonce_job_${i}.sh
# done

# for i in {0..10}; do
#     echo "Submitting job for tokenize_smolLM2_sent_job_${i}.sh"
#     pjsub tokenize_smolLM2_sent_job_${i}.sh
# done

# for i in {0..10}; do
#     echo "Submitting job for tokenize_olmo2_job_${i}.sh"
#     pjsub olmo2-sml/job_${i}.sh
# done
for i in {0..10}; do
    echo "Submitting job for tokenize_qwen_job_${i}.sh"
    pjsub qwen2.5-0.5B-sml/job_${i}.sh
done