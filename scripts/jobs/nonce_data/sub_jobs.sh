#! /bin/bash 

for i in {0..10}; do
    echo "Submitting job for generate_nonce_data_job_${i}.sh"
    pjsub generate_nonce_data_job_${i}.sh
done