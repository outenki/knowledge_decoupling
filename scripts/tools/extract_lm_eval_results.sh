#!/bin/bash

MODEL_NAME=$1

EVAL_SET=blimp
METRIC="acc,none"
echo ">>> extracting results from $MODEL_NAME $EVAL_SET $METRIC"
JSON_FILE=$(ls -tr "$PROJECT_BASE_PATH/output/${MODEL_NAME}/eval/$EVAL_SET/"*.json | tail -1)
uv run python extract_lm_eval_results.py $JSON_FILE $METRIC

EVAL_SET=qa
METRIC="acc,norm"
echo ">>> extracting results from $MODEL_NAME $EVAL_SET $METRIC"
JSON_FILE=$(ls -tr "$PROJECT_BASE_PATH/output/${MODEL_NAME}/eval/$EVAL_SET/"*.json | tail -1)
uv run python extract_lm_eval_results.py $JSON_FILE $METRIC

EVAL_SET=context_qa
METRIC="none"
echo ">>> extracting results from $MODEL_NAME $EVAL_SET $METRIC"
JSON_FILE=$(ls -tr "$PROJECT_BASE_PATH/output/${MODEL_NAME}/eval/$EVAL_SET/"*.json | tail -1)
uv run python extract_lm_eval_results.py $JSON_FILE $METRIC