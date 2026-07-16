#!/bin/bash

MODEL_NAME=$1

EVAL_SET=blimp
METRIC="acc,none"
echo ">>> extracting blimp results from $MODEL_NAME $EVAL_SET $METRIC"
JSON_FILE=$(ls -tr "$PROJECT_BASE_PATH/output/${MODEL_NAME}/eval/$EVAL_SET/"*.json | tail -1)
echo ">>> JSON_FILE: $JSON_FILE"
uv run python extract_lm_eval_results.py $JSON_FILE $METRIC

EVAL_SET=qa
METRIC="none"
echo ">>> extracting qa results from $MODEL_NAME $EVAL_SET $METRIC"
JSON_FILE=$(ls -tr "$PROJECT_BASE_PATH/output/${MODEL_NAME}/eval/$EVAL_SET/"*.json | tail -1)
echo ">>> JSON_FILE: $JSON_FILE"
uv run python extract_lm_eval_results.py $JSON_FILE $METRIC

EVAL_SET=context_qa/ewok
METRIC="acc,none"
echo ">>> extracting ${EVAL_SET} results from $MODEL_NAME $EVAL_SET $METRIC"
JSON_FILE=$(ls -tr "$PROJECT_BASE_PATH/output/${MODEL_NAME}/eval/$EVAL_SET/results_"*.json | tail -1)
echo ">>> JSON_FILE: $JSON_FILE"
uv run python extract_lm_eval_results.py $JSON_FILE $METRIC avg

EVAL_SET=context_qa/squadv2
METRIC="f1,none"
echo ">>> extracting ${EVAL_SET} results from $MODEL_NAME $EVAL_SET $METRIC"
JSON_FILE=$(ls -tr "$PROJECT_BASE_PATH/output/${MODEL_NAME}/eval/$EVAL_SET/results_"*.json | tail -1)
echo ">>> JSON_FILE: $JSON_FILE"
uv run python extract_lm_eval_results.py $JSON_FILE $METRIC

EVAL_SET=context_qa/squadv2
METRIC="f1,none"
echo ">>> extracting ${EVAL_SET} results from $MODEL_NAME $EVAL_SET $METRIC"
JSON_FILE=$(ls -tr "$PROJECT_BASE_PATH/output/${MODEL_NAME}-sft_mix_train/eval/$EVAL_SET/results_"*.json | tail -1)
echo ">>> JSON_FILE: $JSON_FILE"
uv run python extract_lm_eval_results.py $JSON_FILE $METRIC

EVAL_SET=context_qa/squadv2
METRIC="f1,none"
echo ">>> extracting ${EVAL_SET} results from $MODEL_NAME $EVAL_SET $METRIC"
JSON_FILE=$(ls -tr "$PROJECT_BASE_PATH/output/${MODEL_NAME}-sft_squadv2_train/eval/$EVAL_SET/results_"*.json | tail -1)
echo ">>> JSON_FILE: $JSON_FILE"
uv run python extract_lm_eval_results.py $JSON_FILE $METRIC

EVAL_SET=context_qa/boolq_local
METRIC="none"
echo ">>> extracting ${EVAL_SET} results from $MODEL_NAME $EVAL_SET $METRIC"
JSON_FILE=$(ls -tr "$PROJECT_BASE_PATH/output/${MODEL_NAME}/eval/$EVAL_SET/results_"*.json | tail -1)
echo ">>> JSON_FILE: $JSON_FILE"
uv run python extract_lm_eval_results.py $JSON_FILE $METRIC

EVAL_SET=context_qa/boolq_local
METRIC="none"
echo ">>> extracting ${EVAL_SET} results from $MODEL_NAME $EVAL_SET $METRIC"
JSON_FILE=$(ls -tr "$PROJECT_BASE_PATH/output/${MODEL_NAME}-sft_mix_train/eval/$EVAL_SET/results_"*.json | tail -1)
echo ">>> JSON_FILE: $JSON_FILE"
uv run python extract_lm_eval_results.py $JSON_FILE $METRIC

EVAL_SET=context_qa/boolq_local
METRIC="none"
echo ">>> extracting ${EVAL_SET} results from $MODEL_NAME $EVAL_SET $METRIC"
JSON_FILE=$(ls -tr "$PROJECT_BASE_PATH/output/${MODEL_NAME}-sft_boolq_local_train/eval/$EVAL_SET/results_"*.json | tail -1)
echo ">>> JSON_FILE: $JSON_FILE"
uv run python extract_lm_eval_results.py $JSON_FILE $METRIC

EVAL_SET=context_qa/triviaqa_rc_context
METRIC="none"
echo ">>> extracting ${EVAL_SET} results from $MODEL_NAME $EVAL_SET $METRIC"
JSON_FILE=$(ls -tr "$PROJECT_BASE_PATH/output/${MODEL_NAME}/eval/$EVAL_SET/results_"*.json | tail -1)
echo ">>> JSON_FILE: $JSON_FILE"
uv run python extract_lm_eval_results.py $JSON_FILE $METRIC

EVAL_SET=context_qa/triviaqa_rc_context
METRIC="none"
echo ">>> extracting ${EVAL_SET} results from $MODEL_NAME $EVAL_SET $METRIC"
JSON_FILE=$(ls -tr "$PROJECT_BASE_PATH/output/${MODEL_NAME}-sft_mix_train/eval/$EVAL_SET/results_"*.json | tail -1)
echo ">>> JSON_FILE: $JSON_FILE"
uv run python extract_lm_eval_results.py $JSON_FILE $METRIC

EVAL_SET=context_qa/triviaqa_rc_context
METRIC="none"
echo ">>> extracting ${EVAL_SET} results from $MODEL_NAME $EVAL_SET $METRIC"
JSON_FILE=$(ls -tr "$PROJECT_BASE_PATH/output/${MODEL_NAME}-sft_triviaqa_rc_context_train/eval/$EVAL_SET/results_"*.json | tail -1)
echo ">>> JSON_FILE: $JSON_FILE"
uv run python extract_lm_eval_results.py $JSON_FILE $METRIC