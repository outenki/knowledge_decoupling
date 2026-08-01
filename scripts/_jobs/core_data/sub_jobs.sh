#! /bin/bash 

# pjsub generate_qa_triviaqa_core_job.sh
# pjsub generate_qa_squadv2_core_job.sh
# pjsub generate_qa_google_boolq_core_job.sh
for i in {0..10}; do
    echo "Submitting job for core_smolLM2_job_${i}.sh"
    pjsub core_smolLM2_job_${i}.sh
done
