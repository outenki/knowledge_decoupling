#! /bin/bash 

for i in {0..20}; do
    echo "Submitting job for filter_dataset_${i}.sh"
    pjsub filter_dataset_job_${i}.sh
done