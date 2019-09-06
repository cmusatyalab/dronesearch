#!/usr/bin/env bash
#
# Drone Search
#
# A computer vision pipeline for live video search on drone video
# feeds leveraging edge servers.
#
# Copyright (C) 2018-2019 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
set -ex

die() { echo "$@" 1>&2 ; exit 1; }
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

dataset_ids=(
    'raft'
    'elephant'
)

for DATASET in "${dataset_ids[@]}"
do
    echo "========================================"
    echo "Starting training $DATASET"
    echo "========================================"
    cd $DIR
    DATASET_DIR=$DIR/$DATASET
    RESIZED_LONG_EDGE=448
    RESIZED_SHORT_EDGE=224
    TILE_WIDTH=224
    TILE_HEIGHT=224
    LAST_LAYER_MAX_STEP=20000
    ALL_LAYER_MAX_STEP=40000
    PRETRAINED_CHECKPOINT_DIR=/home/junjuew/mobisys18/pretrained_models/mobilenet_ckpt

    TRAIN_PYENV_PATH=$DIR/../pyenv
    EVAL_PYENV_PATH=/home/junjuew/cv/tf1.3-cpu-vector-env

    ANNOTATION_DIR=${DATASET_DIR}/annotations
    RESIZED_IMAGE_DIR=${DATASET_DIR}/images_${RESIZED_LONG_EDGE}_${RESIZED_SHORT_EDGE}
    TILE_ANNOTATION_DIR=${DATASET_DIR}/classification_${RESIZED_LONG_EDGE}_${RESIZED_SHORT_EDGE}_${TILE_WIDTH}_${TILE_HEIGHT}_annotations
    EXPERIMENT_DIR=${DATASET_DIR}/experiments/classification_${RESIZED_LONG_EDGE}_${RESIZED_SHORT_EDGE}_${TILE_WIDTH}_${TILE_HEIGHT}_extra_negative
    LAST_LAYER_TRAIN_DIR="${EXPERIMENT_DIR}/logs_last_layer_only_${LAST_LAYER_MAX_STEP}"
    ALL_LAYER_TRAIN_DIR="${EXPERIMENT_DIR}/logs_all_layers_${ALL_LAYER_MAX_STEP}"

    source $TRAIN_PYENV_PATH/bin/activate
    cd mobilenet/research/slim
    cat > scripts/finetune_mobilenet_v1_on_twoclass.shrc <<EOL
TRAIN_LAST_LAYER=false
TRAIN_ALL_LAYER=true
PRETRAINED_CHECKPOINT_DIR="${PRETRAINED_CHECKPOINT_DIR}"
DATASET_DIR="${EXPERIMENT_DIR}"
LAST_LAYER_MAX_STEP=${LAST_LAYER_MAX_STEP}
ALL_LAYER_MAX_STEP=${ALL_LAYER_MAX_STEP}
LAST_LAYER_TRAIN_DIR="${LAST_LAYER_TRAIN_DIR}"
ALL_LAYER_TRAIN_DIR="${ALL_LAYER_TRAIN_DIR}"
MAX_GPU_MEMORY_USAGE=1.0
EOL

    bash scripts/finetune_mobilenet_v1_on_twoclass.sh &
    trainer_bgid=$!
    deactivate

    source $EVAL_PYENV_PATH/bin/activate
    bash scripts/continuous_eval_mobilenet_v1_on_twoclass.sh -t ${ALL_LAYER_TRAIN_DIR} -d ${EXPERIMENT_DIR} -s validation &
    bash scripts/continuous_eval_mobilenet_v1_on_twoclass.sh -t ${ALL_LAYER_TRAIN_DIR} -d ${EXPERIMENT_DIR} -s train &
    deactivate

    echo "Launched Training for $DATASET"
    wait ${trainer_bgid}
done
