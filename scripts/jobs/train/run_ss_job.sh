#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o log/run_ss.log


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/run/train

mkdir -p log

sh run_ss.sh google_re_conflict_short_context
sh run_ss.sh google_re_conflict_long_context
