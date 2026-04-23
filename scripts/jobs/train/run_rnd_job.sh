#!/bin/bash
#PBS -q lg
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o log/run_rnd.log


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/run/train

mkdir -p log

sh run_rnd.sh google_re_conflict_short_context
sh run_rnd.sh google_re_conflict_long_context

