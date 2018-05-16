#!/usr/bin/env bash
# The plan to add more points into figure 14.
# The event recall is acquired by uniformly sample. We can find at which threshold the event recall changed from
# early discard
# Current dataset only use the dataset's own test videos. We can use the test videos from other datasets as well

set -e

BASE_DIR=/home/junjuew/mobisys18/scripts
MY_BASE_DIR=/home/junjuew/work/drone-scalable-search/experiments/jitl
DATASET=${DATASET:-raft}
DATASETS=(
"okutama"
"stanford"
"raft"
"elephant"
)
early_discard_plotted_threshold_and_recall_file="/home/junjuew/mobisys18/experiments/early_discard_extra_negative\
/plotted_event_recall.pkl"

for DATASET in ${DATASETS[@]}; do

EXP_DATASET_DIR=${MY_BASE_DIR}/${DATASET}
mkdir -p ${EXP_DATASET_DIR}
mkdir -p ${EXP_DATASET_DIR}/log

# Generate data
if false; then
        printf "Generating JITL input data into $s\n" "${EXP_DATASET_DIR}"
        python jitl_data.py make_jitl_dataframe \
            --base_dir ${BASE_DIR}  \
            --dataset ${DATASET}    \
            --output_file ${EXP_DATASET_DIR}/jitl_input.pkl
fi

# get dnn thresholds that are interesting (event recall changes)
if false; then
    python jitl_data.py make_jitl_dnn_threshold \
        --early_discard_plotted_threshold_and_recall_file ${early_discard_plotted_threshold_and_recall_file} \
        --dataset ${DATASET} \
        --output_file ${EXP_DATASET_DIR}/jitl_dnn_threshold.pkl | tee \
        ${EXP_DATASET_DIR}/log/dnn_threshold_${DATASET}.log
fi

# Run simulation
# Interesting cutoff range:
# Okutama, elephant: 0.90~0.99; stanford, raft: 0.8~1.0; stanford
if false; then
    mkdir -p ${EXP_DATASET_DIR}/log
    python jitl_test.py eval_jit_svm_on_dataset \
        --jit_data_file ${EXP_DATASET_DIR}/jitl_input.pkl \
        --output_file ${EXP_DATASET_DIR}/jitl_result.pkl \
        --dnn_threshold_input_file ${EXP_DATASET_DIR}/jitl_dnn_threshold.pkl \
        --svm_cutoff "[0.1, 0.3, 0.5, 0.7, 0.9]" \
        | tee ${EXP_DATASET_DIR}/log/eval_jit_svm_${DATASET}.log
#        --dnn_cutoff_start 80 \
#        --dnn_cutoff_end 100 \
#        --dnn_cutoff_step 2 \
fi

# Plot
if true; then
    echo ""
    echo "Plotting frames vs event recall"
    python jitl_plot.py frames_vs_event_recall \
        --base_dir ${BASE_DIR} \
        --dataset ${DATASET} \
        --jitl_result_file ${EXP_DATASET_DIR}/jitl_result.pkl \
        --savefig ${EXP_DATASET_DIR}/fig-jitl-${DATASET}-eventrecall.pdf
fi

done; exit

# Generate max-pooled data. Same as sampling an image from a sliding window
# it generates a file of the same schema as the previous command.
# change the jit_data_file name in following commands to use this one's output.
# span/stride sec -> frame: 0.2/0.1->6/3; 0.07/0.04->2/1; 0.17/0.08->2/1 1.0/0.5->30/15
# span=1,stride=0.5 hurts recall
# select only 1 image for svm with the highest proba in span.
if false; then
    python jitl_test.py max_pooling_on_dataset \
        --jit_data_file ${EXP_DATASET_DIR}/jitl_input.pkl \
        --mp_span_secs 0.2 \
        --mp_stride_secs 0.1 \
        --output_file ${EXP_DATASET_DIR}/jitl_input-maxpool.pkl
fi

#    echo ""
#    echo "Plotting frames vs DNN cutoff"
#    python jitl_plot.py frames_vs_dnn_cutoff \
#        --jitl_result_file ${EXP_DATASET_DIR}/jitl_result.pkl \
#        --savefig ${MY_BASE_DIR}/experiments/jitl/figure/fig-jitl-${DATASET}-frame.pdf

#    echo ""
#    echo "Plotting event recall vs DNN cutoff"
#    python jitl_plot.py event_recall_vs_dnn_cutoff \
#        --base_dir ${BASE_DIR} \
#        --dataset ${DATASET} \
#        --jitl_result_file ${EXP_DATASET_DIR}/jitl_result.pkl \
#        --savefig ${MY_BASE_DIR}/experiments/jitl/figure/fig-jitl-${DATASET}-eventrecall-cutoff.pdf

echo "Done with dataset ${DATASET}!"
