#! /bin/bash

SML_DIR=$PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/no_warmup/sml/SmolLM2-135M-20B-sml-bs4096
MASK_DIR=$PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/no_warmup/sml_mask/SmolLM2-135M-20B-sml_mask-bs4096
RND_ID_DIR=$PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/no_warmup/sml_rnd_id/SmolLM2-135M-20B-rnd_id-bs4096
RND_DIR=$PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/no_warmup/rnd/rnd

# SFT
# TASK=google_boolq
# METRIC=acc
# uv run $PROJECT_BASE_PATH/scripts/eval/bootstrap_resampling.py \
#     -sml $SML_DIR-sft_${TASK}_train/eval/${TASK} \
#     -core $MASK_DIR-sft_${TASK}_train/eval/${TASK} \
#     -m $METRIC \
#     -o $MASK_DIR-sft_${TASK}_train/eval/${TASK}/bootstrap_resampling_results.json
# uv run $PROJECT_BASE_PATH/scripts/eval/bootstrap_resampling.py \
#     -sml $SML_DIR-sft_${TASK}_train/eval/${TASK} \
#     -core $RND_ID_DIR-sft_${TASK}_train/eval/${TASK} \
#     -m $METRIC \
#     -o $RND_ID_DIR-sft_${TASK}_train/eval/${TASK}/bootstrap_resampling_results.json
# uv run $PROJECT_BASE_PATH/scripts/eval/bootstrap_resampling.py \
#     -sml $SML_DIR-sft_${TASK}_train/eval/${TASK} \
#     -core $RND_DIR-sft_${TASK}_train/eval/${TASK} \
#     -m $METRIC \
#     -o $RND_DIR-sft_${TASK}_train/eval/${TASK}/bootstrap_resampling_results.json

# w/o SFT
# TASK=lambada_openai
# METRIC=acc
# echo
# echo "Bootstrap resampling for ${MASK_DIR}/eval/${TASK}..."
# uv run $PROJECT_BASE_PATH/scripts/eval/bootstrap_resampling.py \
#     -sml $SML_DIR/eval/${TASK} \
#     -core $MASK_DIR/eval/${TASK} \
#     -m $METRIC \
#     -o $MASK_DIR/eval/${TASK}/bootstrap_resampling_results.json
# echo
# echo "Bootstrap resampling for ${RND_ID_DIR}/eval/${TASK}..."
# uv run $PROJECT_BASE_PATH/scripts/eval/bootstrap_resampling.py \
#     -sml $SML_DIR/eval/${TASK} \
#     -core $RND_ID_DIR/eval/${TASK} \
#     -m $METRIC \
#     -o $RND_ID_DIR/eval/${TASK}/bootstrap_resampling_results.json
# echo
# echo "Bootstrap resampling for ${RND_DIR}/eval/${TASK}..."
# uv run $PROJECT_BASE_PATH/scripts/eval/bootstrap_resampling.py \
#     -sml $SML_DIR/eval/${TASK} \
#     -core $RND_DIR/eval/${TASK} \
#     -m $METRIC \
#     -o $RND_DIR/eval/${TASK}/bootstrap_resampling_results.json

# google re
METRIC=acc
TASK=google_re_mix_short
echo
echo "Bootstrap resampling for ${MASK_DIR}/eval/${TASK}..."
uv run $PROJECT_BASE_PATH/scripts/eval/bootstrap_resampling.py \
    -sml $SML_DIR/SmolLM2-135M-20B-sml-bs4096-sft_google_re_mix_short_train/eval/${TASK} \
    -core $MASK_DIR/SmolLM2-135M-20B-sml_mask-bs4096-sft_google_re_mix_short_train/eval/${TASK} \
    -m $METRIC \
    -o $MASK_DIR/SmolLM2-135M-20B-sml_mask-bs4096-sft_google_re_mix_short_train/eval/${TASK}/bootstrap_resampling_results.json
TASK=google_re_mix_conflict_short
echo
echo "Bootstrap resampling for ${MASK_DIR}/eval/${TASK}..."
uv run $PROJECT_BASE_PATH/scripts/eval/bootstrap_resampling.py \
    -sml $SML_DIR/SmolLM2-135M-20B-sml-bs4096-sft_google_re_mix_short_train/eval/${TASK} \
    -core $MASK_DIR/SmolLM2-135M-20B-sml_mask-bs4096-sft_google_re_mix_short_train/eval/${TASK} \
    -m $METRIC \
    -o $MASK_DIR/SmolLM2-135M-20B-sml_mask-bs4096-sft_google_re_mix_short_train/eval/${TASK}/bootstrap_resampling_results.json