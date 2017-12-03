#!/usr/bin/env bash

BASE_DIR=/home/zf/opt/drone-scalable-search/processed_dataset
DATASET=raft

if true; then
    python jitl_data.py make_jitl_dataframe
        --base_dir ${BASE_DIR}  \
        --dataset ${DATASET}
        --output_file ${BASE_DIR}/${DATASET}/experiments/jitl/tile_label_and_proba.csv
fi
