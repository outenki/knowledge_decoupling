#!/bin/bash
#PBS -q c30897g
#PBS -l select=1:ngpus=4
#PBS -l walltime=160:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/train_Llama3.2_1B_sml_mask_43.log
#PBS -N "tr_43_lsm"



SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/../../common/run_llama3_2_1b.sh" Llama3.2_1B-sml-mask 43
