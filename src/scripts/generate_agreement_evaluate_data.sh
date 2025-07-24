#!/bin/bash
uv run \
python src/generate_agreement_evaluate_data.py \
    -dp data/preprocessed  \
    -dn test \
    -lf local \
    -o data/evaluation/agreement_evaluate_data
