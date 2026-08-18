#! /bin/bash

# for i in {1..10}
for i in 0
do
    echo "Running filter_dataset_by_replace_ratio.py - part_$i"
    uv run python filter_dataset_by_replace_ratio.py \
        /home/pj24001974/ku50001571/projects/knowledge_decoupling/data/SmolLM2-135M-20B/core_ent/parts/part_$i \
        /home/pj24001974/ku50001571/projects/knowledge_decoupling/data/SmolLM2-135M-20B/core_ent/rp5/part_$i
done