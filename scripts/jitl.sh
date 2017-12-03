#!/usr/bin/env bash

BASE_DIR=/home/zf/opt/drone-scalable-search
DATASET=raft

if false; then
    python jitl_data.py make_jitl_dataframe \
        --base_dir ${BASE_DIR}/processed_dataset  \
        --dataset ${DATASET}    \
        --output_file ${BASE_DIR}/processed_dataset/${DATASET}/experiments/jitl/jitl_input.pkl
fi

if false; then
    python jitl_test.py eval_jit_svm_on_dataset \
        --jit_data_file ${BASE_DIR}/processed_dataset/${DATASET}/experiments/jitl/jitl_input.pkl \
        --output_file ${BASE_DIR}/processed_dataset/${DATASET}/experiments/jitl/jitl_result.pkl \
        --dnn_cutoff_list "[0.1, 0.9]"
fi

if true; then
    python ${BASE_DIR}/experiments/jitl/plot.py frames_vs_dnn_cutoff \
        --jitl_data_file ${BASE_DIR}/processed_dataset/${DATASET}/experiments/jitl/jitl_input.pkl \
        --jitl_result_file ${BASE_DIR}/processed_dataset/${DATASET}/experiments/jitl/jitl_result.pkl \
        --savefig ${BASE_DIR}/experiments/jitl/figure/fig-${DATASET}-frame.pdf
fi