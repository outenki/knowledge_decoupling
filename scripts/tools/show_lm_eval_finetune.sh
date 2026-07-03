EVAL_PATH=$1

sft=sft_based_squad_train
echo
echo ">>>> Processing $sft"
mv $EVAL_PATH-$sft/eval/based_suqad_local/ $EVAL_PATH-$sft/eval/based_squad_local/ 
p=$EVAL_PATH-$sft/eval/based_squad_local
JSON_FILE=$(ls -tr "$p/"*.json | tail -1)
echo $JSON_FILE
uv run python extract_lm_eval_results.py $JSON_FILE "contains,none"

sft=sft_squad_v2_train
echo
echo ">>>> Processing $sft"
mv $EVAL_PATH-$sft/eval/suqad_v2/ $EVAL_PATH-$sft/eval/squad_v2/ 
p=$EVAL_PATH-$sft/eval/squad_v2/
JSON_FILE=$(ls -tr "$p/"*.json | tail -1)
echo $JSON_FILE
uv run python extract_lm_eval_results.py $JSON_FILE "f1,none"

sft=sft_race_train
echo
echo ">>>> Processing $sft"
p=$EVAL_PATH-$sft/eval/race_local
JSON_FILE=$(ls -tr "$p/"*.json | tail -1)
echo $JSON_FILE
uv run python extract_lm_eval_results.py $JSON_FILE "acc,none"

sft=sft_boolq_train
echo
echo ">>>> Processing $sft"
p=$EVAL_PATH-$sft/eval/boolq_local
JSON_FILE=$(ls -tr "$p/"*.json | tail -1)
echo $JSON_FILE
uv run python extract_lm_eval_results.py $JSON_FILE "acc,none"

