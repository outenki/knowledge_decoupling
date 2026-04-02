#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=10:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o log/run_hf.log


source $HOME/.zshrc
cd /lustre1/work/c30897/wtq/projects/knowledge_decoupling/scripts/run

mkdir -p log

sh run_hf.sh google_re_conflict_short_context
sh run_hf.sh google_re_conflict_long_context
