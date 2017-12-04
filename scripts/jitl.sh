#!/usr/bin/env bash

set -e

BASE_DIR=/home/junjuew/mobisys18/scripts
MY_BASE_DIR=/home/zf/opt/drone-scalable-search
DATASET=${DATASET:-raft}

mkdir -p ${MY_BASE_DIR}/processed_dataset/${DATASET}/experiments/jitl

# Generate data
if false; then
    python jitl_data.py make_jitl_dataframe \
        --base_dir ${BASE_DIR}  \
        --dataset ${DATASET}    \
        --output_file ${MY_BASE_DIR}/processed_dataset/${DATASET}/experiments/jitl/jitl_input.pkl
fi

# Run simulation
if true; then
    mkdir -p ${MY_BASE_DIR}/processed_dataset/${DATASET}/experiments/jitl/log
    python jitl_test.py eval_jit_svm_on_dataset \
        --jit_data_file ${MY_BASE_DIR}/processed_dataset/${DATASET}/experiments/jitl/jitl_input.pkl \
        --output_file ${MY_BASE_DIR}/processed_dataset/${DATASET}/experiments/jitl/jitl_result.pkl \
        --dnn_cutoff_start 0 \
        --dnn_cutoff_end 100 \
        --dnn_cutoff_step 10 \
        | tee ${MY_BASE_DIR}/processed_dataset/${DATASET}/experiments/jitl/log/eval_jit_svm_${DATASET}.log
fi

# Plot
if true; then
#    echo ""
#    echo "Plotting frames vs DNN cutoff"
#    python jitl_plot.py frames_vs_dnn_cutoff \
#        --jitl_data_file ${MY_BASE_DIR}/processed_dataset/${DATASET}/experiments/jitl/jitl_input.pkl \
#        --jitl_result_file ${MY_BASE_DIR}/processed_dataset/${DATASET}/experiments/jitl/jitl_result.pkl \
#        --savefig ${MY_BASE_DIR}/experiments/jitl/figure/fig-jitl-${DATASET}-frame.pdf

    echo ""
    echo "Plotting frames vs event recall"
    python jitl_plot.py frames_vs_event_recall \
        --base_dir ${BASE_DIR} \
        --dataset ${DATASET} \
        --jitl_result_file ${MY_BASE_DIR}/processed_dataset/${DATASET}/experiments/jitl/jitl_result.pkl \
        --savefig ${MY_BASE_DIR}/experiments/jitl/figure/fig-jitl-${DATASET}-eventrecall.pdf

#    echo ""
#    echo "Plotting event recall vs DNN cutoff"
#    python jitl_plot.py event_recall_vs_dnn_cutoff \
#        --base_dir ${BASE_DIR} \
#        --dataset ${DATASET} \
#        --jitl_result_file ${MY_BASE_DIR}/processed_dataset/${DATASET}/experiments/jitl/jitl_result.pkl \
#        --savefig ${MY_BASE_DIR}/experiments/jitl/figure/fig-jitl-${DATASET}-eventrecall-cutoff.pdf
fi

echo "Done with dataset ${DATASET}!"