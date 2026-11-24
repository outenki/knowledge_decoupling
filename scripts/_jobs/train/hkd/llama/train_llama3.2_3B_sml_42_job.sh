#!/bin/bash
#PBS -q c30897g
#PBS -l select=1:ngpus=4
#PBS -l walltime=160:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o "logs/train_llama3.2_3B_sml_42.log"
#PBS -N "tr_42_3B"

# Environment
source "$HOME/.zshrc"

SIF=/lustre1/work/c30897/wtq/env/singularity/axolotl-ngc2605.sif
VENV_PATH=/opt/axolotl-venv
export HF_HOME="$PROJECT_BASE_PATH/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$PROJECT_BASE_PATH/.cache"

cd "$PROJECT_BASE_PATH/src/train"

# Optional: WandB offline mode
# export WANDB_MODE=offline

# Run training inside Singularity
singularity exec \
    --fakeroot \
    --nv \
    --bind "$PROJECT_BASE_PATH:$PROJECT_BASE_PATH" \
    --bind "$VENV_PATH:$VENV_PATH" \
    --pwd "$PROJECT_BASE_PATH/src/train" \
    "$SIF" \
    "$VENV_PATH/bin/python" train.py \
        --config-name Llama3.2_3B-sml \
        base.path="$PROJECT_BASE_PATH" \
        seed=42