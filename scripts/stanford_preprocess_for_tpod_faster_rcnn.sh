# SCENE_FRAME_SEQUENCE_DIR="frames/nexus"
# SCENE_ANNOTATION_DIR="annotations_car_only/nexus"
# TRAIN_LIST='["video1","video2","video5","video8","video9","video11"]'
# OUTPUT_DIR="experiments/car_only_0"
# # test list is "video4","video7"

# python gen_experiments.py gen_car_only_experiments $SCENE_FRAME_SEQUENCE_DIR $SCENE_ANNOTATION_DIR $TRAIN_LIST $OUTPUT_DIR

BASE_DIR="/home/junjuew/mobisys18/processed_dataset/stanford_campus"

SCENE_FRAME_SEQUENCE_DIR="${BASE_DIR}/frames/nexus"
SCENE_ANNOTATION_DIR="${BASE_DIR}/annotations_car_only/nexus"
TRAIN_LIST='["video4","video7"]'
OUTPUT_DIR="${BASE_DIR}/experiments/car_only_1/validation"
# test list is "video4","video7"

python preprocess_for_tpod_faster_rcnn.py gen_car_only_experiments $SCENE_FRAME_SEQUENCE_DIR $SCENE_ANNOTATION_DIR $TRAIN_LIST $OUTPUT_DIR
