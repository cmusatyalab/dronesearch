#!/usr/bin/env bash
set -ex

die() { echo "$@" 1>&2 ; exit 1; }

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_DIR=$DIR/okutama
IMAGE_WIDTH=224
IMAGE_HEIGHT=224
TILE_WIDTH=224
TILE_HEIGHT=224
SAMPLE_NUM_PER_CLASS=2000
LAST_LAYER_MAX_STEP=5000
ALL_LAYER_MAX_STEP=10000
PRETRAINED_CHECKPOINT_DIR=/mnt/junjuew/mobisys18/pretrained_models/mobilenet_ckpt

# TRAIN_PYENV_PATH=$DIR/../pyenv
# EVAL_PYENV_PATH=/home/junjuew/cv/tf1.3-cpu-vector-env
TRAIN_PYENV_PATH=/home/junjuew/cv/tf1.3-gpu-env
EVAL_PYENV_PATH=/mnt/junjuew/cv/tf1.3-cpu-vector-env

ANNOTATION_DIR=${DATASET_DIR}/annotations
RESIZED_IMAGE_DIR=${DATASET_DIR}/images_${IMAGE_WIDTH}_${IMAGE_HEIGHT}
TILE_ANNOTATION_DIR=${DATASET_DIR}/classification_${TILE_WIDTH}_${TILE_HEIGHT}_annotations
EXPERIMENT_DIR=${DATASET_DIR}/experiments/classification_${TILE_WIDTH}_${TILE_HEIGHT}
LAST_LAYER_TRAIN_DIR="${EXPERIMENT_DIR}/logs_last_layer_only_${LAST_LAYER_MAX_STEP}"
ALL_LAYER_TRAIN_DIR="${EXPERIMENT_DIR}/logs_all_layers_${ALL_LAYER_MAX_STEP}"

# For 3840x2160 only
# DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# DATASET_DIR=$DIR/okutama
# IMAGE_WIDTH=224
# IMAGE_HEIGHT=224
# TILE_WIDTH=224
# TILE_HEIGHT=224
# SAMPLE_NUM_PER_CLASS=2000
# LAST_LAYER_MAX_STEP=5000
# ALL_LAYER_MAX_STEP=10000
# PRETRAINED_CHECKPOINT_DIR=/home/junjuew/mobisys18/pretrained_models/mobilenet_ckpt

# TRAIN_PYENV_PATH=/home/junjuew/mobisys18/pyenv
# EVAL_PYENV_PATH=/home/junjuew/cv/tf1.3-cpu-vector-env

# ANNOTATION_DIR=${DATASET_DIR}/annotations
# RESIZED_IMAGE_DIR=${DATASET_DIR}/images_3840_2160_down_224_224
# TILE_ANNOTATION_DIR=${DATASET_DIR}/classification_3840_2160_down_224_224_annotations
# EXPERIMENT_DIR=${DATASET_DIR}/experiments/classification_3840_2160_down_224_224_annotations
# LAST_LAYER_TRAIN_DIR="${EXPERIMENT_DIR}/logs_last_layer_only_${LAST_LAYER_MAX_STEP}"
# ALL_LAYER_TRAIN_DIR="${EXPERIMENT_DIR}/logs_all_layers_${ALL_LAYER_MAX_STEP}"


# Resize if needed
echo "Make sure you resize images to integer multiple of tile width and height"
echo "Not doing resizing"
echo "========================================"

echo "Generating Tile Annotations to ${TILE_ANNOTATION_DIR}"
mkdir -p ${TILE_ANNOTATION_DIR}
python annotation.py get_tile_classification_annotation ${ANNOTATION_DIR} ${IMAGE_WIDTH} ${IMAGE_HEIGHT} ${TILE_WIDTH} ${TILE_HEIGHT} ${TILE_ANNOTATION_DIR} | tee ${TILE_ANNOTATION_DIR}/stats.txt

echo "Sample Tile Images for Training to ${EXPERIMENT_DIR}"
mkdir -p ${EXPERIMENT_DIR}
ln -s ${RESIZED_IMAGE_DIR} ${EXPERIMENT_DIR}/photos
python preprocess.py sample_okutama_frames ${TILE_ANNOTATION_DIR} ${SAMPLE_NUM_PER_CLASS} ${EXPERIMENT_DIR} train &
sleep 10
python preprocess.py sample_okutama_frames ${TILE_ANNOTATION_DIR} ${SAMPLE_NUM_PER_CLASS} ${EXPERIMENT_DIR} test &
wait

echo "Going into mobilenet dir to launch training"
cd mobilenet/research/slim
python -m datasets/convert_twoclass_tile run --dataset_dir=${EXPERIMENT_DIR} --mode=train --tile_width=${TILE_WIDTH} --tile_height=${TILE_HEIGHT} 2>&1 &
sleep 10
python -m datasets/convert_twoclass_tile run --dataset_dir=${EXPERIMENT_DIR} --mode=test --tile_width=${TILE_WIDTH} --tile_height=${TILE_HEIGHT} 2>&1 &
wait

# To be cont.
cat > scripts/finetune_mobilenet_v1_on_okutama.shrc <<EOL
TRAIN_LAST_LAYER=true
TRAIN_ALL_LAYER=true
PRETRAINED_CHECKPOINT_DIR="${PRETRAINED_CHECKPOINT_DIR}"
DATASET_DIR="${EXPERIMENT_DIR}"
LAST_LAYER_MAX_STEP=${LAST_LAYER_MAX_STEP}
ALL_LAYER_MAX_STEP=${ALL_LAYER_MAX_STEP}
LAST_LAYER_TRAIN_DIR="${LAST_LAYER_TRAIN_DIR}"
ALL_LAYER_TRAIN_DIR="${ALL_LAYER_TRAIN_DIR}"
MAX_GPU_MEMORY_USAGE=1.0
EOL

source $TRAIN_PYENV_PATH/bin/activate
bash scripts/finetune_mobilenet_v1_on_okutama.sh &
deactivate

source $EVAL_PYENV_PATH/bin/activate
bash scripts/continuous_eval_mobilenet_v1_on_okutama.sh -t ${ALL_LAYER_TRAIN_DIR} -d ${EXPERIMENT_DIR} -s validation &
bash scripts/continuous_eval_mobilenet_v1_on_okutama.sh -t ${ALL_LAYER_TRAIN_DIR} -d ${EXPERIMENT_DIR} -s test &
bash scripts/continuous_eval_mobilenet_v1_on_okutama.sh -t ${ALL_LAYER_TRAIN_DIR} -d ${EXPERIMENT_DIR} -s train &

echo "Launching Training"
wait

