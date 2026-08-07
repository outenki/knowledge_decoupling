DATA_NAME=$1 
BASE_PATH=$PROJECT_BASE_PATH/input/evaluate_data/json

echo ">>> $DATA_NAME"
for split in train validation test; do
    uv run python remove_core_angle.py $BASE_PATH/${DATA_NAME}_core_angle/$split.json $BASE_PATH/${DATA_NAME}_core_rnd/$spli.json 
done