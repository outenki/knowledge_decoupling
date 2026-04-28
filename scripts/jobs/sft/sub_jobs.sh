# pjsub evaluate_hf_job_commonsense_qa.sh
# pjsub evaluate_hf_job_arc_easy.sh
# pjsub evaluate_hf_job_arc_challenge.sh
# pjsub evaluate_hf_job_qasc.sh
pjsub run_hf_job_commonsense_qa.sh
pjsub run_hf_job_arc_easy.sh
pjsub run_hf_job_arc_challenge.sh
pjsub run_hf_job_qasc.sh
# pjsub run_hf_gpt2_job_commonsense_qa.sh
# pjsub run_hf_gpt2_job_arc_easy.sh
# pjsub run_hf_gpt2_job_arc_challenge.sh
# pjsub run_hf_gpt2_job_qasc.sh

watch -d pjstat