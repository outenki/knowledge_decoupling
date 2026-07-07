#! /bin/bash 

# for i in {0..10}; do
#     echo "Submitting job for tokenize_smolLM2_job_${i}.sh"
#     pjsub tokenize_smolLM2_job_${i}.sh
# done

# for i in {0..10}; do
#     echo "Submitting job for tokenize_gpt2_job_${i}.sh"
#     pjsub tokenize_gpt2_job_${i}.sh
# done

# for i in {0..10}; do
#     echo "Submitting job for tokenize_llama_job_${i}.sh"
#     pjsub tokenize_llama_job_${i}.sh
# done

for i in {0..10}; do
    echo "Submitting job for tokenize_olme2_job_${i}.sh"
    pjsub olme2/job_${i}.sh
done
for i in {0..10}; do
    echo "Submitting job for tokenize_qwen_job_${i}.sh"
    pjsub qwen2.5-0.5B/job_${i}.sh
done