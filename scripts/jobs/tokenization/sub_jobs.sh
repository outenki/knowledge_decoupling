#! /bin/bash 

for i in {0..10}; do
    echo "Submitting job for tokenize_smolLM2_job_${i}.sh"
    pjsub tokenize_smolLM2_job_${i}.sh
done