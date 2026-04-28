#! /bin/bash 

for i in {5703578..5703599}; do
    echo "Deleting job for filter_dataset_${i}.sh"
    pjdel $i
done