#! /bin/bash 

for i in {0..10}; do
    echo "Submitting job for core_smolLM2_job_${i}.sh"
    pjsub core_smolLM2_job_${i}.sh
done
