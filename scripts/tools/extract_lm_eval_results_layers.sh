#!/bin/bash

MODEL_NAME=$1
EVAL_SET=$2
LAYER=$3
METRIC=${4:-acc,none}

JSON_FILE=$(ls -tr "$PROJECT_BASE_PATH/output/${MODEL_NAME}/layers_${LAYER}/base/eval/$EVAL_SET/layers_${LAYER}__base/"*.json | tail -1)
uv run python extract_lm_eval_results.py $JSON_FILE $METRIC